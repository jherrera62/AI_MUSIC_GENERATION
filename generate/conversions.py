import mido
import numpy as np
import os

# Set the lower and upper bounds of the MIDI note range
lower_note_bound = 24
upper_note_bound = 102
note_boundary = upper_note_bound - lower_note_bound


def midi_conversion(midi_file, squash=True, span=note_boundary):
    # Load the MIDI file
    midi = mido.MidiFile(midi_file)
    # Calculate the ticks per quarter note and eighth note
    quarter_note_ticks = (midi.ticks_per_beat / 4)
    eighth_note_ticks = (midi.ticks_per_beat / 8)
    time = 0
    state_matrix = []
    positions = [0 for _ in range(len(midi.tracks))]
    note_state = [[0, 0] for _ in range(span)]
    iterate = True
    remaining_time = [track[0].time for track in midi.tracks]
    state_matrix.append(note_state)

    while iterate:
        # Add a new state to the state matrix at each eighth note interval
        if time % quarter_note_ticks == eighth_note_ticks:
            prev_state = note_state
            note_state = [[prev_state[x][0], 0] for x in range(span)]
            state_matrix.append(note_state)

        for i in range(len(remaining_time)):
            if not iterate:
                break
            # This loop processes events for a single time step.
            # extracts the MIDI track and position for the current iteration.
            while remaining_time[i] == 0:
                track = midi.tracks[i]
                position = positions[i]
                event = track[position]
                # checks if the note falls within the specified range 
                if event.type == 'note_on':
                    if lower_note_bound <= event.note < upper_note_bound:
                        if event.velocity == 0:
                            note_state[event.note - lower_note_bound] = [0, 0]  # Set note as off
                        else:
                            note_state[event.note - lower_note_bound] = [1, 1]  # Set note as on
                # updates the state of the corresponding note in the note_state array to indicate that the note is turned off.
                elif event.type == 'note_off':
                    if lower_note_bound <= event.note < upper_note_bound:
                        note_state[event.note - lower_note_bound] = [0, 0]  # Set note as off
                # checks if the time signature is non-standard (not 2/4 or 4/4)
                elif event.type == 'time_signature':
                    if event.numerator not in (2, 4):
                        out = state_matrix
                        iterate = False
                        break
                # updates the remaining time for the current track and advances the position within the track.
                try:
                    remaining_time[i] = track[position + 1].time
                    positions[i] += 1
                except IndexError:
                    remaining_time[i] = None
            # If the remaining time for the current track is not None, it decrements the remaining time by 1.
            if remaining_time[i] is not None:
                remaining_time[i] -= 1
        # checks if all remaining times are None for all tracks. If so, it breaks out of the loop.
        if all(t is None for t in remaining_time):
            break

        time += 1

    # Convert the state matrix to a suitable format for further processing
    state_array = np.array(state_matrix)
    state_matrix = np.hstack((state_array[:, :, 0], state_array[:, :, 1]))
    state_matrix = np.asarray(state_matrix).tolist()
    return state_matrix


def state_matrix_conversion(state_matrix, name="example", span=note_boundary):
    # Converts the state_matrix variable into a NumPy array. 
    state_matrix = np.array(state_matrix)
    # Checks if the shape of the state_matrix array has three dimensions.
    if not len(state_matrix.shape) == 3:
        #  If the state_matrix is not a 3D array, this step stacks the existing 2D state_matrix along the third dimension
        state_matrix = np.dstack((state_matrix[:, :span], state_matrix[:, span:]))
    #  Converts the state_matrix back to a NumPy array
    state_matrix = np.asarray(state_matrix)

    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    tickscale = 70
    # This variable keeps track of the previous time in the state matrix. 
    previous_time = 0
    # represents the previous state of the notes in the state matrix
    previous_state = [[0, 0] for _ in range(span)]
    # iterates over each time step and state in the state_matrix
    for time, state in enumerate(state_matrix + [previous_state[:]]):
        #  keep track of notes that need to be turned off or turned on
        off = []
        on = []
        for i in range(span):
            current_note_state = state[i]
            previous_note_state = previous_state[i]
            # Check for note-off events by comparing the previous and current note states.
            if previous_note_state[0] == 1:
                if current_note_state[0] == 0:
                    off.append(i)
                elif current_note_state[1] == 1:
                    off.append(i)
                    on.append(i)
            # Check for note-on events
            elif current_note_state[0] == 1:
                on.append(i)
        # Add note-off and note-on messages to the MIDI track
        for note in off:
            track.append(mido.Message('note_off', note=note + lower_note_bound, time=(time - previous_time) * tickscale))
            previous_time = time
        # Adds MIDI messages for turning on the notes in the on list
        for note in on:
            track.append(mido.Message('note_on', note=note + lower_note_bound, velocity=80, time=(time - previous_time) * tickscale))
            previous_time = time
        # Updates the previous_state to the current state, preparing for the next iteration
        previous_state = state

    # Add end-of-track message to the MIDI track
    track.append(mido.MetaMessage('end_of_track'))
    # Save the file to the folder named generated_songs
    midi_file_path = os.path.join('generated_songs', "{}.mid".format(name))
    # Save the generated MIDI file
    midi_file.save(midi_file_path)
