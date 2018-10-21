# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 19:24:48 2016

@author: wuyiming
"""

from madmom.utils import midi
from madmom.utils.midi import NoteEvent,NoteOffEvent,NoteOnEvent
import numpy as np

class CustomMIDIFile(midi.MIDIFile):
    def notes(self, note_time_unit='s'):
        # list for all notes
        notes = []
        # dictionaries for storing the last onset and velocity per pitch
        note_onsets = {}
        note_velocities = {}
        for track in self.tracks:
            # get a list with note events
            note_events = [e for e in track.events if (isinstance(e, NoteEvent) and not (e.channel in [9,10]))]
            # process all events
            tick = 0
            for e in note_events:
                if tick > e.tick:
                    raise AssertionError('note events must be sorted!')

                is_note_on = isinstance(e, NoteOnEvent)
                is_note_off = isinstance(e, NoteOffEvent)
                # if it's a note on event with a velocity > 0,
                if is_note_on and e.velocity > 0:
                    # save the onset time and velocity
                    note_onsets[e.pitch] = e.tick
                    note_velocities[e.pitch] = e.velocity
                # if it's a note off event or a note on with a velocity of 0,
                elif is_note_off or (is_note_on and e.velocity == 0):
                    # the old velocity must be greater 0
                    if note_velocities[e.pitch] <= 0:
                        raise AssertionError('note velocity must be positive')
                    if note_onsets[e.pitch] > e.tick:
                        raise AssertionError('note duration must be positive')
                    # append the note to the list
                    notes.append((note_onsets[e.pitch], e.pitch,
                                  e.tick - note_onsets[e.pitch],
                                  note_velocities[e.pitch]))
                else:
                    raise TypeError('unexpected NoteEvent')
                tick = e.tick

        # sort the notes and convert to numpy array
        notes.sort()
        notes = np.asarray(notes, dtype=np.float)

        # convert onset times and durations from ticks to a meaningful unit
        # and return the notes
        if note_time_unit == 's':
            return self._note_ticks_to_seconds(notes)
        elif note_time_unit == 'b':
            return self._note_ticks_to_beats(notes)
        else:
            raise ValueError("note_time_unit must be either 's' (seconds) or "
                             "'b' (beats), not %s." % note_time_unit)