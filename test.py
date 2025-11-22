import mido
import numpy as np
import json

def diagnose_midi(midi_path):
    """Analyze a MIDI file to see what's actually in it"""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {midi_path}")
    print(f"{'='*60}\n")
    
    try:
        mid = mido.MidiFile(midi_path)
        print(f"✓ MIDI file loaded successfully")
        print(f"  Type: {mid.type}")
        print(f"  Ticks per beat: {mid.ticks_per_beat}")
        print(f"  Total tracks: {len(mid.tracks)}")
        print(f"  Length: {mid.length:.2f} seconds\n")
        
        for i, track in enumerate(mid.tracks):
            print(f"Track {i}: {len(track)} messages")
            
            note_on_count = 0
            note_off_count = 0
            tempo = None
            current_tick = 0
            notes_info = []
            
            for msg in track:
                current_tick += msg.time
                
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    print(f"  Tempo: {tempo} microseconds/beat ({60000000/tempo:.1f} BPM)")
                
                elif msg.type == 'note_on' and msg.velocity > 0:
                    note_on_count += 1
                    time_seconds = mido.tick2second(current_tick, mid.ticks_per_beat, tempo if tempo else 500000)
                    notes_info.append({
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'time': time_seconds,
                        'tick': current_tick
                    })
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note_off_count += 1
            
            print(f"  Note ON events: {note_on_count}")
            print(f"  Note OFF events: {note_off_count}")
            
            if notes_info:
                print(f"\n  First 5 notes:")
                for j, note_info in enumerate(notes_info[:5]):
                    freq = 440 * (2 ** ((note_info['note'] - 69) / 12))
                    print(f"    {j+1}. MIDI Note: {note_info['note']:3d} ({freq:7.2f} Hz), "
                          f"Velocity: {note_info['velocity']:3d}, "
                          f"Time: {note_info['time']:7.3f}s (tick {note_info['tick']})")
            else:
                print(f"  ⚠ WARNING: NO NOTES FOUND!")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR reading MIDI file: {e}")
        return False

def create_test_midi(output_path='test.mid'):
    """Create a simple test MIDI file to verify playback works"""
    print(f"\n{'='*60}")
    print("CREATING TEST MIDI FILE")
    print(f"{'='*60}\n")
    
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('set_tempo', tempo=500000))
    
    # Create a simple scale: C4, D4, E4, F4, G4
    notes = [60, 62, 64, 65, 67]  # MIDI note numbers
    
    for i, note in enumerate(notes):
        # Note ON at beat i
        track.append(mido.Message('note_on', note=note, velocity=80, time=480 if i > 0 else 0))
        # Note OFF half a beat later
        track.append(mido.Message('note_off', note=note, velocity=80, time=240))
    
    mid.save(output_path)
    print(f"✓ Test MIDI saved to {output_path}")
    print(f"  Contains {len(notes)} notes (C-D-E-F-G scale)")
    print(f"  Duration: ~2.5 seconds")
    print(f"\nTry playing this with: fluidsynth -a alsa -m alsa_seq -l -i /usr/share/sounds/sf2/FluidR3_GM.sf2 {output_path}")
    print(f"Or: timidity {output_path}")
    
    return output_path

def analyze_training_data(json_path='train.json'):
    """Check if training data looks reasonable"""
    print(f"\n{'='*60}")
    print(f"ANALYZING TRAINING DATA: {json_path}")
    print(f"{'='*60}\n")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        notes = data.get('notes_corrected', [])
        print(f"✓ Total notes in dataset: {len(notes)}")
        
        if len(notes) == 0:
            print("✗ ERROR: No notes in dataset!")
            return False
        
        # Analyze the data
        start_times = [n[0] for n in notes if len(n) >= 3]
        end_times = [n[1] for n in notes if len(n) >= 3]
        frequencies = [n[2] for n in notes if len(n) >= 3]
        
        durations = [e - s for s, e in zip(start_times, end_times)]
        
        print(f"\nStart times: {min(start_times):.3f}s to {max(start_times):.3f}s")
        print(f"Durations: {min(durations):.3f}s to {max(durations):.3f}s (avg: {np.mean(durations):.3f}s)")
        print(f"Frequencies: {min(frequencies):.1f}Hz to {max(frequencies):.1f}Hz")
        
        print(f"\nFirst 5 notes:")
        for i in range(min(5, len(notes))):
            n = notes[i]
            if len(n) >= 3:
                dur = n[1] - n[0]
                print(f"  {i+1}. Start: {n[0]:.3f}s, End: {n[1]:.3f}s, Freq: {n[2]:.1f}Hz, Duration: {dur:.3f}s")
        
        # Check for issues
        if min(durations) < 0.01:
            print(f"\n⚠ WARNING: Very short durations detected (min: {min(durations):.6f}s)")
        if max(frequencies) > 5000 or min(frequencies) < 20:
            print(f"\n⚠ WARNING: Unusual frequency range detected")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ ERROR: File not found: {json_path}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def generate_wav_from_notes(notes, output_path='output.wav', sample_rate=44100):
    """
    Generate a WAV file directly from note data
    notes: list of [freq, start, end]
    """
    print(f"\n{'='*60}")
    print(f"GENERATING WAV FILE: {output_path}")
    print(f"{'='*60}\n")
    
    try:
        import scipy.io.wavfile as wavfile
    except ImportError:
        print("✗ scipy not installed. Install with: pip install scipy")
        return False
    
    if not notes:
        print("✗ No notes provided!")
        return False
    
    # Find total duration
    max_end = max(note[2] for note in notes)
    total_samples = int(max_end * sample_rate) + sample_rate  # +1 second padding
    
    audio = np.zeros(total_samples)
    
    print(f"Generating {len(notes)} notes...")
    for i, (freq, start, end) in enumerate(notes):
        if end <= start or freq < 20 or freq > 5000:
            print(f"  Skipping invalid note {i+1}")
            continue
        
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        duration_samples = end_sample - start_sample
        
        if duration_samples <= 0:
            continue
        
        # Generate sine wave
        t = np.linspace(0, (end - start), duration_samples, False)
        tone = np.sin(2 * np.pi * freq * t)
        
        # Apply envelope (fade in/out to avoid clicks)
        envelope_len = min(int(0.01 * sample_rate), duration_samples // 4)
        envelope = np.ones(duration_samples)
        envelope[:envelope_len] = np.linspace(0, 1, envelope_len)
        envelope[-envelope_len:] = np.linspace(1, 0, envelope_len)
        
        tone = tone * envelope * 0.3  # Scale amplitude
        
        # Add to audio buffer
        if start_sample + len(tone) <= len(audio):
            audio[start_sample:start_sample + len(tone)] += tone
    
    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"✓ WAV file saved to {output_path}")
    print(f"  Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"\nPlay with: aplay {output_path} (Linux) or any audio player")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LSTM MUSIC GENERATION DIAGNOSTICS")
    print("="*60)
    
    # 1. Check training data
    analyze_training_data('train.json')
    
    # 2. Create a test MIDI file
    test_midi = create_test_midi('test_simple.mid')
    
    # 3. Check if predicted MIDI exists and diagnose it
    import os
    if os.path.exists('predicted_sequence.mid'):
        diagnose_midi('predicted_sequence.mid')
        
        # Try to convert MIDI to WAV for easier playback
        print("\n" + "="*60)
        print("ATTEMPTING TO CONVERT MIDI TO WAV")
        print("="*60)
        
        try:
            # Read the predicted notes
            mid = mido.MidiFile('predicted_sequence.mid')
            notes = []
            tempo = 500000
            current_time = 0
            active_notes = {}
            
            for msg in mid.tracks[0]:
                current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start = active_notes[msg.note]
                        end = current_time
                        freq = 440 * (2 ** ((msg.note - 69) / 12))
                        notes.append([freq, start, end])
                        del active_notes[msg.note]
            
            if notes:
                generate_wav_from_notes(notes, 'predicted_sequence.wav')
            else:
                print("⚠ No notes found in MIDI file to convert")
                
        except Exception as e:
            print(f"✗ Error converting MIDI to WAV: {e}")
    else:
        print("\n⚠ predicted_sequence.mid not found. Run the training script first!")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Play test_simple.mid to verify your MIDI player works")
    print("2. Check the analysis above for any warnings")
    print("3. If predicted_sequence.wav was created, try playing that")
    print("4. Share the output above if you need more help!")
