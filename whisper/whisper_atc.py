# Load model directly
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# pipe = pipeline(
#   "automatic-speech-recognition",
#   model="openai/whisper-tiny",
#   chunk_length_s=60,
#   device=torch.device("mps"),
# )


# Load the pipeline for speech-to-text, specifying the model and its processor
# speech_recognition = pipeline(
# 	"automatic-speech-recognition",
# 	model="jlvdoorn/whisper-large-v3-atco2-asr-atcosim",
# 	device=torch.device('mps'))


# speech_recognition = pipeline(
# 	"automatic-speech-recognition",
# 	model="openai/whisper-tiny",
# 	return_timestamps=True,
# 	device=torch.device('mps'))
#
# # Example usage (assuming you have an audio file)
# results = speech_recognition("atc_train.mp3")['text']
# print(results)


import whisper

# Load the model and the processor
import whisper

model = whisper.load_model("large-v3")
general = ['Air Traffic Control communications']
nato = [
	'Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf',
	'Hotel', 'India', 'Juliett', 'Kilo', 'Lima', 'Mike', 'November',
	'Oscar', 'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango', 'Uniform',
	'Victor', 'Whiskey', 'Xray', 'Yankee', 'Zulu'
]
atc_common_words = [
	"acknowledge", "affirmative", "altitude", "approach", "apron", "arrival",
	"bandbox", "base", "bearing", "cleared", "climb", "contact", "control",
	"crosswind", "cruise", "descend", "departure", "direct", "disregard",
	"downwind", "estimate", "final", "flight", "frequency", "go around",
	"heading", "hold", "identified", "immediate", "information", "instruct",
	"intentions", "land", "level", "maintain", "mayday", "message", "missed",
	"navigation", "negative", "obstruction", "option", "orbit", "pan-pan",
	"pattern", "position", "proceed", "radar", "readback", "received",
	"report", "request", "required", "runway", "squawk", "standby", "takeoff",
	"taxi", "threshold", "traffic", "transit", "turn", "vector", "visual",
	"waypoint", "weather", "wilco", "wind", "with you", "altitude", "speed",
	"heavy", "light", "medium", "emergency", "fuel", "squawk", "identifier",
	"limit", "monitor", "notice", "operation", "permission", "relief",
	"route", "signal", "stand", "system", "terminal", "test", "track",
	"understand", "verify", "vertical", "warning", "zone", "no", "yes", "unable"
]
more_phrases = ['ATC',
                'Pilot',
                'Call sign',
                'Altitude',
                'Heading',
                'Speed',
                'Climb to',
                'Descend to',
                'Maintain',
                'Approach',
                'Tower',
                'Ground',
                'Runway',
                'Taxi',
                'Takeoff',
                'Landing',
                'Flight level',
                'Squawk',
                'Radar contact',
                'Traffic',
                'Hold short',
                'Cleared for',
                'Go around',
                'Read back',
                'Roger',
                'Wilco',
                'Affirmative',
                'Negative',
                'Standby',
                'Mayday',
                'Pan-pan',
                'Flight plan',
                'Visibility',
                'Weather',
                'Wind',
                'Gusts',
                'Turbulence',
                'Icing conditions',
                'Deicing',
                'Instrument Landing System (ILS)',
                'Visual Flight Rules (VFR)',
                'Instrument Flight Rules (IFR)',
                'No-fly zone',
                'Restricted airspace',
                'Flight path',
                'Direct route',
                'Vector',
                'Frequency change',
                'Handoff',
                'Final approach',
                'Initial climb to',
                'Contact approach',
                'Squawk ident',
                'Flight information region (FIR)',
                'Control zone',
                'Terminal control area (TMA)',
                'Standard instrument departure (SID)',
                'Standard terminal arrival route (STAR)',
                'Missed approach',
                'Holding pattern',
                'Minimum safe altitude',
                'Transponder',
                'Traffic alert and collision avoidance system (TCAS)',
                'Reduce speed to',
                'Increase speed to',
                'Flight conditions',
                'Clear of conflict',
                'Resume own navigation',
                'Request altitude change',
                'Request route change',
                'Flight visibility',
                'Ceiling',
                'Severe weather',
                'Convective SIGMET',
                'AIRMET',
                'NOTAM',
                'QNH',
                'QFE',
                'Transition altitude',
                'Transition level',
                'No significant change (NOSIG)',
                'Temporary flight restriction (TFR)',
                'Special use airspace',
                'Military operation area (MOA)',
                'Instrument approach procedure (IAP)',
                'Visual approach',
                'Non-directional beacon (NDB)',
                'VHF omnidirectional range (VOR)',
                'Automatic terminal information service (ATIS)',
                'Pushback',
                'Engine start clearance',
                'Line up and wait',
                'Unicom',
                'Cross runway',
                'Backtrack',
                'Departure frequency',
                'Arrival frequency',
                'Go-ahead',
                'Hold position',
                'Check gear down']

collated_list = general + nato + atc_common_words + more_phrases

collated_list_string = ' '.join(collated_list)

result = model.transcribe("atc_train.mp3", verbose=True, language="en", prompt=collated_list_string)
# print(result["text"])
