import os
import re
import ast
from datetime import datetime

# Specify the directory path
transcriptions_dir = "./transcriptions"

# Check if the directory exists
if not os.path.exists(transcriptions_dir):
	# If the directory does not exist, create it
	os.makedirs(transcriptions_dir)


def find_latest_transcription(directory):
	# Regex pattern for matching the filename
	pattern = re.compile(r'transcription_(\d{2})(\d{2})(\d{2})\.txt')
	latest_file = None
	latest_date = None

	for filename in os.listdir(directory):
		match = pattern.match(filename)
		if match:
			# Extract day, month, year from the filename
			day, month, year = match.groups()
			file_date = datetime.strptime(f'20{year}{month}{day}', '%Y%m%d')

			# Update the latest file based on date
			if not latest_date or file_date > latest_date:
				latest_date = file_date
				latest_file = filename

	return latest_file if latest_file else None


# Attempt to find the latest transcription file
latest_transcription_file = find_latest_transcription(transcriptions_dir)

if latest_transcription_file:
	# Full path for the latest file including the directory
	file_path = os.path.join(transcriptions_dir, latest_transcription_file)
	try:
		with open(file_path, "r") as stored_result:
			# If the file exists and is opened successfully, read the content
			temp_result = stored_result.read()
			result = ast.literal_eval(temp_result)
			# result = dict(result)
		print('\033[92mTranscription located:\033[0m')
		print(result['text'])

	except FileNotFoundError:
		print("File not found, although it was expected to exist.")
else:
	# No transcription file matching the pattern was found
	print('\033[91mNo matching transcription files found.\033[0m')
	# If the file does not exist, execute the transcription process and create the file
	temp_result = str(model.transcribe("./content/conference_call.mp3", verbose=True,
	                              language="en", prompt=collated_list_string))
	file_path = os.path.join(transcriptions_dir, f'transcription_{datetime.now().strftime("%y%m%d")}.txt')
	with open(file_path, "w") as f:
		f.write(temp_result)
	result = ast.literal_eval(temp_result)
	print(f'\033[92mTranscription completed and saved at {file_path}.\033[0m')

per_line = []
for segment in result['segments']:
	text_to_append = segment['text']
	text_to_append = text_to_append[1:]
	per_line.append(text_to_append)

per_line

# experiment 1

from collections import deque
STRIDE = 2
NUMBER_OF_CHUNKS = 60

def chunk_with_stride_and_indices(initial_list: list, stride: int, number_of_chunks: int):
	N = len(initial_list)
	chunk_indices = list(range(0, N))

	# Create initial chunks
	initial_chunks = [initial_list[i * number_of_chunks:(i + 1) * number_of_chunks] for i in
	                  range((N + number_of_chunks - 1) // number_of_chunks)]
	initial_chunk_indices = [chunk_indices[i * number_of_chunks:(i + 1) * number_of_chunks] for i in
	                         range((N + number_of_chunks - 1) // number_of_chunks)]

	stride_chunks = []
	stride_chunk_indices = []  # Track indices for each strided chunk

	for chunk_index in range(len(initial_chunks)):
		current_indices = deque(initial_chunk_indices[chunk_index])

		if chunk_index == 0:
			stride_chunks.append(initial_chunks[chunk_index])
			stride_chunk_indices.append(list(current_indices))
		elif 0 < chunk_index < len(initial_chunks) - 1:
			current_chunk = deque(initial_chunks[chunk_index])

			# Retrieve elements and their indices from the previous chunk to perform backwards stride
			previous_chunk_elements = initial_chunks[chunk_index - 1][-stride:]
			previous_chunk_elements.sort(reverse=True)
			previous_indices = initial_chunk_indices[chunk_index - 1][-stride:]
			previous_indices.sort(reverse=True)

			for past_element, past_index in zip(previous_chunk_elements, previous_indices):
				current_chunk.appendleft(past_element)
				current_indices.appendleft(past_index)

			future_chunk_elements = initial_chunks[chunk_index + 1][:stride]
			future_indices = initial_chunk_indices[chunk_index + 1][:stride]

			for future_element, future_index in zip(future_chunk_elements, future_indices):
				current_chunk.append(future_element)
				current_indices.append(future_index)

			stride_chunks.append(list(current_chunk))
			stride_chunk_indices.append(list(current_indices))

		elif chunk_index == len(initial_chunks) - 1:
			current_chunk = deque(initial_chunks[chunk_index])
			previous_chunk_elements = initial_chunks[chunk_index - 1][-stride:]
			previous_chunk_elements.sort(reverse=True)
			previous_indices = initial_chunk_indices[chunk_index - 1][-stride:]
			previous_indices.sort(reverse=True)

			for past_element, past_index in zip(previous_chunk_elements, previous_indices):
				current_chunk.appendleft(past_element)
				current_indices.appendleft(past_index)

			stride_chunks.append(list(current_chunk))
			stride_chunk_indices.append(list(current_indices))
		else:
			raise ValueError("Invalid chunk index")

	print("Strided Chunks:", stride_chunks)
	print("Strided Chunk Indices:", stride_chunk_indices)
	return stride_chunks, stride_chunk_indices


strided_chunks, strided_chunk_indices = chunk_with_stride_and_indices(per_line, STRIDE,NUMBER_OF_CHUNKS)

def chunk_with_stride_and_indices(initial_list: list, stride: int, number_of_chunks: int):
	N = len(initial_list)
	chunk_indices = list(range(0, N))

	# Calculate the size of each chunk
	chunk_size = (N + number_of_chunks - 1) // number_of_chunks

	# Create initial chunks
	initial_chunks = [initial_list[i * chunk_size:(i + 1) * chunk_size] for i in range(number_of_chunks)]
	initial_chunk_indices = [chunk_indices[i * chunk_size:(i + 1) * chunk_size] for i in range(number_of_chunks)]

	stride_chunks = []
	stride_chunk_indices = []  # Track indices for each strided chunk

	for chunk_index in range(len(initial_chunks)):
		current_chunk = deque(initial_chunks[chunk_index])
		current_indices = deque(initial_chunk_indices[chunk_index])

		if chunk_index > 0:  # For all chunks except the first, prepend elements from the previous chunk
			previous_chunk_elements = initial_chunks[chunk_index - 1][-stride:]
			previous_indices = initial_chunk_indices[chunk_index - 1][-stride:]

			for past_element, past_index in zip(reversed(previous_chunk_elements), reversed(previous_indices)):
				current_chunk.appendleft(past_element)
				current_indices.appendleft(past_index)

		if chunk_index < len(initial_chunks) - 1:  # For all chunks except the last, append elements from the next chunk
			future_chunk_elements = initial_chunks[chunk_index + 1][:stride]
			future_indices = initial_chunk_indices[chunk_index + 1][:stride]

			for future_element, future_index in zip(future_chunk_elements, future_indices):
				current_chunk.append(future_element)
				current_indices.append(future_index)

		stride_chunks.append(list(current_chunk))
		stride_chunk_indices.append(list(current_indices))

	print("Strided Chunks:", stride_chunks)
	print("Strided Chunk Indices:", stride_chunk_indices)
	return stride_chunks, stride_chunk_indices


strided_chunks, strided_chunk_indices = chunk_with_stride_and_indices(per_line, STRIDE, NUMBER_OF_CHUNKS)

def chunk_with_stride_and_indices(initial_list: list, stride: int, number_of_chunks: int):
	stride -= 1
	N = len(initial_list)

	# Calculate base chunk size without considering stride for simplicity
	base_chunk_size = (N + number_of_chunks - 1) // number_of_chunks

	# Prepare initial chunks without stride
	initial_chunks = [initial_list[i * base_chunk_size:(i + 1) * base_chunk_size] for i in range(number_of_chunks)]
	initial_chunk_indices = [list(range(i * base_chunk_size, min((i + 1) * base_chunk_size, N))) for i in
	                         range(number_of_chunks)]

	stride_chunks = []
	stride_chunk_indices = []

	for i in range(number_of_chunks):
		# Calculate the effective start and end, incorporating stride where applicable
		start = max(0, i * base_chunk_size - stride)
		end = min(N, (i + 1) * base_chunk_size + stride if i < number_of_chunks - 1 else N)

		# Slice the original list and indices accordingly
		current_chunk = initial_list[start:end]
		current_indices = list(range(start, end))

		stride_chunks.append(current_chunk)
		stride_chunk_indices.append(current_indices)

	return stride_chunks, stride_chunk_indices


strided_chunks, strided_chunk_indices = chunk_with_stride_and_indices(per_line, stride=STRIDE, number_of_chunks=NUMBER_OF_CHUNKS)
# Demonstrating output
for i, (chunk, indices) in enumerate(zip(strided_chunks, strided_chunk_indices), start=1):
	# print(f"Chunk {i}: {chunk}")
	print(f"Indices {i}: {indices}")

#experiment 2
# Strided Chunk-based diarization without speaker labels

speaker_labels = []

for strided_chunk in strided_chunks:
	strided_formatted = '\n'.join(strided_chunk)
	diarization = client.chat.completions.create(
		model=DEPLOYMENT,
		messages=[
			{"role": "system",
			 "content": "You are a linguistics expert with 50 years of experience. You will be given a list of sentences seperated by newlines, and you are to assign the Speaker label to each sentence PER line. I.e. Given the prompt, you will return me: ['Speaker 1', 'Speaker 2', 'Speaker 2']. There is a possibility that a speaker may speak for more than 1 line at time."},
			{"role": "user",
			 "content": f"Here is the list of sentences: {strided_formatted}. You will diarize ALL the sentences in the list. JUST RETURN ME THE LIST. You WILL ensure that you labeled EVERY line. You will return me: ['Speaker 1', 'Speaker 2', 'Speaker 2']"}
		],
		max_tokens=2000,
		stream=False,
		temperature=0.5,
	)
	end_result = diarization.choices[0].message.content
	speaker_labels.append(end_result)

print(speaker_labels)

# experiment 3
#remove empty lists
strided_chunk_indices = [ele for ele in strided_chunk_indices if ele != []]
strided_chunks = [ele for ele in strided_chunks if ele != []]
import pandas as pd
import numpy as np

# Assuming 'per_line' and 'strided_chunk_indices' are defined elsewhere in the script
comparison_df = pd.DataFrame({'original': per_line})

# Use numpy to efficiently calculate the min and max values for DataFrame index
min_val = np.min([min(sublist) for sublist in strided_chunk_indices])
max_val = np.max([max(sublist) for sublist in strided_chunk_indices])

# Initialize the DataFrame with the correct index range
strided_chunk_df = pd.DataFrame(index=np.arange(min_val, max_val + 1))

# Populate the DataFrame with strided chunk data
for i, sublist in enumerate(strided_chunk_indices):
    # Direct assignment to the DataFrame using loc for precise index matching
    strided_chunk_df.loc[sublist, f'strided_chunk_{i}'] = strided_chunks[i]

# Combine the initial comparison DataFrame with the newly created strided chunk DataFrame
combined_df = pd.concat([comparison_df, strided_chunk_df], axis=1)

combined_df
import ast
import pandas as pd


class TextDiarizer:
	def __init__(self, client, deployment):
		self.client = client
		self.deployment = deployment

	def diarize_chunk(self, sentences, prev_labels_info=None):
		"""Diarize a chunk of text, optionally using information from previous labels."""
		system_message = "You are a linguistics expert with 100 years of experience. You will be given a transcription of a MEETING between an unknown number of speakers, and you are to assign the Speaker label to each sentence PER line. I.e. Given the prompt, you will return me: ['Speaker 1', 'Speaker 2', 'Speaker 2']. There is a possibility that a speaker may speak for more than 1 line at time. You will DO YOUR JOB WELL."

		if prev_labels_info:
			prev_speaker_labels = list(prev_labels_info.values())
			sentences_dict = prev_speaker_labels[0]
			speaker_labels_dict = prev_speaker_labels[1]

			# creating {'sentence':speaker_label} dictionary
			sentence_speaker_mapping = {value: speaker_labels_dict[key] for key, value in sentences_dict.items()}

			user_message = f"Here are the previous exchanges RIGHT before this followed by their respective speaker(s):\n{sentence_speaker_mapping} \n Here is the list of sentences:\n{sentences}\nNote that this contains the previous exchanges as well. I MUST RECEIVE ALL {len(sentences)} exchanges. JUST RETURN ME THE LIST."
		else:
			user_message = f"Here is the list of sentences: \n{sentences}. \nThere are {len(sentences)} exchanges. You will diarize ALL the sentences in the list. You WILL ensure that you label ALL {len(sentences)} lines. JUST RETURN ME THE LIST."
		print('user message:', user_message + '\n\n')
		diarization = self.client.chat.completions.create(
			model=self.deployment,
			messages=[
				{"role": "system", "content": system_message},
				{"role": "user", "content": user_message}
			],
			max_tokens=2500,
			stream=False,
			temperature=0.2,
		)
		return ast.literal_eval(diarization.choices[0].message.content)

	def get_stride_info(self, chunk_df, total_label_list, chunk_number, stride):
		"""Retrieve and label the information for a given stride."""
		column_name = f'strided_chunk_{chunk_number}'
		chunk_df[column_name] = chunk_df[column_name].replace('nan', np.nan)

		_ = chunk_df[column_name].dropna()
		stride_df = pd.DataFrame(_.tail(stride))
		previous_speaker_labels = total_label_list[-1][-stride:]
		stride_df['speaker_labels'] = previous_speaker_labels
		return stride_df.to_dict()

	def label_aware(self, stride, number_of_chunks, combined_chunk_df):
		total_label_list = []

		# Process the first chunk
		combined_chunk_df['strided_chunk_0'] = combined_chunk_df['strided_chunk_0'].replace('nan', np.nan)
		first_chunk = combined_chunk_df['strided_chunk_0'].dropna().tolist()
		speaker_labels = self.diarize_chunk(first_chunk)
		total_label_list.append(speaker_labels)

		# Process subsequent chunks
		for chunk_number in range(1, (len(combined_chunk_df.columns) - 1)):
			column_name = f'strided_chunk_{chunk_number}'
			combined_chunk_df[column_name] = combined_chunk_df[column_name].replace('nan', np.nan)
			current_chunk = combined_chunk_df[column_name].dropna(how='any').tolist()
			prev_labels_info = self.get_stride_info(combined_chunk_df, total_label_list, chunk_number - 1, stride)
			speaker_labels = self.diarize_chunk(current_chunk, prev_labels_info)
			total_label_list.append(speaker_labels)

		return total_label_list


diarizer = TextDiarizer(client, DEPLOYMENT)
final_labels = diarizer.label_aware(STRIDE, NUMBER_OF_CHUNKS, combined_df)
print(final_labels)


def final_df(per_line, strided_chunk_indices, final_labels):
	# Assuming 'per_line' and 'strided_chunk_indices' are defined elsewhere in the script
	label_comparison_df = pd.DataFrame({'original': per_line})

	# Use numpy to efficiently calculate the min and max values for DataFrame index
	min_val = np.min([min(sublist) for sublist in strided_chunk_indices])
	max_val = np.max([max(sublist) for sublist in strided_chunk_indices])

	# Initialize the DataFrame with the correct index range
	label_strided_chunk_df = pd.DataFrame(index=np.arange(min_val, max_val + 1))

	# Populate the DataFrame with strided chunk data
	for i, sublist in enumerate(strided_chunk_indices):
		# Direct assignment to the DataFrame using loc for precise index matching
		label_strided_chunk_df.loc[sublist, f'strided_chunk_{i}'] = final_labels[i]

	# Combine the initial comparison DataFrame with the newly created strided chunk DataFrame
	combined_label_df = pd.concat([label_comparison_df, label_strided_chunk_df], axis=1)
	return combined_label_df


final_label_df = final_df(per_line, strided_chunk_indices, final_labels)
final_label_df