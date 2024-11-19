from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import re
import requests
from dateutil import parser
from datetime import datetime
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from google.colab import drive
from transformers import T5Tokenizer, T5ForConditionalGeneration
from difflib import get_close_matches

# Mount Google Drive

# Define the load path from Google Drive

# Load the Roberta tokenizer and model for intent classification
model_path = "/content/drive/MyDrive/fine-tuned-t5"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


intents = [
    "meeting_booking",
    "cab_booking",
    "meeting_cancelling",
    "cab_cancelling",
    "list_cabs_booked",
    "list_meetings_booked",
    "Greetings"
]


# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize FastAPI app
app = FastAPI()

# Define the InputText model for the API input
class InputText(BaseModel):
    text: str

# LanguageTool API for spelling correction
def correct_spelling_with_languagetool(text):
    url = "https://api.languagetool.org/v2/check"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"text": text, "language": "en-US"}
    response = requests.post(url, headers=headers, data=data)
    result = response.json()

    if 'matches' in result and result['matches']:
        corrected_text = text
        for match in result['matches']:
            if 'replacements' in match and match['replacements']:
                replacement = match['replacements'][0]['value']
                incorrect_text = text[match['offset']:match['offset'] + match['length']]
                corrected_text = corrected_text.replace(incorrect_text, replacement)
        return corrected_text
    return text

# Date extraction and conversion
cancellation_pattern = re.compile(r'\b(cancel|unschedule|remove)\b.*\b(meeting|meeting hall)\b', re.IGNORECASE)

def extract_dates(sentence):
    date_pattern = r'\b\d{2}/\d{2}/2024\b'
    matches = re.findall(date_pattern, sentence)
    return matches

def convert_dates(dates):
    converted_dates = []
    for date_str in dates:
        try:
            date_obj = parser.parse(date_str, dayfirst=True)  # Specify dayfirst to align with your date format
            converted_dates.append(date_obj.strftime('%d/%m/%Y'))
        except ValueError:
            # Skip the date if it's invalid
            continue
    return converted_dates

# Unified extraction function for hall names and discussion rooms
def extract_hall_name(input_text):
    hall_names = ["New York", "Mumbai", "Huston", "Amsterdam", "Delhi", "Tokyo", "Chicago"]
    discussion_rooms = ["0a", "0b", "0c", "1a", "1b", "1c", "2a", "2b", "2c"]

    all_locations = hall_names + discussion_rooms

    found_locations = [loc for loc in all_locations if re.search(rf'\b{loc}\b', input_text, re.IGNORECASE)]

    hall_name_error = None
    hall_name = None

    if len(found_locations) > 1:
        hall_name_error = "Error: More than one location provided."
    elif len(found_locations) == 1:
        hall_name = found_locations[0]

    return hall_name, hall_name_error

def check_dropping_points(input_text):
    # List of valid dropping points (case-insensitive matching)
    dropping_points = [
        "Chainsys Company", "Elcot Main Gate", "Madurai Kamaraj College",
        "Nagamalai Puthukottai", "Achampathu", "Kalavasal"
    ]

    # Find all matches in the input text (case-insensitive search)
    found_dropping_points = [point for point in dropping_points if re.search(rf'\b{point}\b', input_text, re.IGNORECASE)]

    # Initialize return values
    dropping_point_error = None
    found_points = None

    # Check if more than 2 dropping points are mentioned
    if len(found_dropping_points) > 2:
        dropping_point_error = "Error: More than two dropping points provided."
    elif found_dropping_points:
        found_points = found_dropping_points

    return found_points, dropping_point_error

# Batch number extraction and validation
def extract_batch_no(input_text):
    batch_patterns = [
        r'\b(?:batch\s*no[-\s]*1|batch[-\s]*1|1(?:st)?[-\s]*batch|batchnumber[-\s]*1)\b',
        r'\b(?:batch\s*no[-\s]*2|batch[-\s]*2|2(?:nd)?[-\s]*batch|batchnumber[-\s]*2)\b'
    ]

    found_batches = []
    batch_no_error = None
    batch_no = None

    for idx, pattern in enumerate(batch_patterns, start=1):
        if re.search(pattern, input_text, re.IGNORECASE):
            found_batches.append(f"batch{idx}")

    if len(found_batches) > 1:
        batch_no_error = "Error: More than one batch has been provided."
    elif len(found_batches) == 1:
        batch_no = found_batches[0]

    return batch_no, batch_no_error

# Cab name extraction and validation
def extract_cab_names(input_text):
    cab_name_pattern = r'\bcab[1-6]\b'
    cab_names = re.findall(cab_name_pattern, input_text, re.IGNORECASE)

    cab_name_error = None
    cab_name = None

    cab_names = list(set([name.lower() for name in cab_names]))  # Normalize to lowercase and remove duplicates

    if len(cab_names) > 1:
        cab_name_error = "Error: More than one cab name provided."
    elif len(cab_names) == 1:
        cab_name = cab_names[0]

    return cab_name, cab_name_error



# Number of persons extraction and validation
def extract_no_of_persons(input_text):
    keywords = [
        "members", "colleagues", "team members", "team leads",
        "people", "participants", "attendees", "individuals"
    ]

    pattern = r'\b(\d+)\s*(?:' + '|'.join(keywords) + r')\b|\b(?:' + '|'.join(keywords) + r')\s*(\d+)\b'

    matches = re.findall(pattern, input_text, re.IGNORECASE)

    no_of_persons_error = None
    no_of_persons = None

    numbers = [int(num) for match in matches for num in match if num]

    if len(numbers) > 1:
        no_of_persons_error = "Error: Multiple values for number of persons provided."
    elif len(numbers) == 1:
        no_of_persons = numbers[0]

    return no_of_persons, no_of_persons_error



# Time extraction and conversion with additional validation
import re
from datetime import datetime

def extract_times(input_text):
    # Regular expression to match times in the `h:mmam/pm` or `h:mmAM/PM` format, and "start to end" ranges
    time_pattern = r'(\b\d{1,2}[:\d{2}]*\s*(?:am|pm|AM|PM)\b|\b\d{1,2}\s*to\s*\d{1,2}\b)'
    
    matches = re.findall(time_pattern, input_text, re.IGNORECASE)
    times = []
    time_error = None

    for match in matches:
        match = match.strip().lower()  # Convert to lowercase to normalize

        if "to" in match:
            # Handle ranges like "9 to 11"
            start, end = match.split("to")
            start_time = datetime.strptime(start.strip(), "%I").strftime("%H:00")
            end_time = datetime.strptime(end.strip(), "%I").strftime("%H:00")
            times.extend([start_time, end_time])
        else:
            # Normalize and parse individual times
            normalized_time = match.strip()
            try:
                time_obj = datetime.strptime(normalized_time, "%I:%M%p")
            except ValueError:
                try:
                    time_obj = datetime.strptime(normalized_time, "%I%p")
                except ValueError:
                    continue
            times.append(time_obj.strftime("%H:%M"))

    start_time, end_time = None, None

    if len(times) > 2:
        time_error = "Error: More than two times provided."
    elif len(times) == 1:
        start_time = times[0]
    elif len(times) == 2:
        start_time = times[0]
        end_time = times[1]

        # Additional validation: ensure start time is less than end time
        if start_time >= end_time:
            time_error = "Error: Time format is not correct as meeting starting time is greater than ending time."

    return start_time, end_time, time_error



def predict(input_text):
    # Focused prompt to instruct the model to classify input into a specific intent
    x=input_text.lower()
    prompt = f"Classify this input into one of the following intents: {', '.join(intents)}. Input: {x}"

    # Tokenize and generate the result
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=546)

    # Decode the result from model output
    classification = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # Match classification to the closest intent from the predefined list
    output = get_close_matches(classification, intents, n=1, cutoff=0.1)  # Adjust cutoff for better matching

    return output[0]



import re

def extract_meeting_booking_id(input_text):
    # Print input text for debugging
    print(f"DEBUG: Input text received: {input_text}")

    # Regular expression to match meeting booking IDs (M followed by exactly 6 digits, case-insensitive)
    meeting_booking_id_pattern = r'[Mm]\d{6}'  # Match 'M' or 'm' followed by 6 digits
    matches = re.findall(meeting_booking_id_pattern, input_text)

    # Debugging: Check what IDs are being found
    print(f"DEBUG: Found meeting booking IDs: {matches}")

    meeting_booking_id = None
    meeting_booking_error = None

    # Handle cases where more than one meeting booking ID is found
    if len(matches) > 1:
        meeting_booking_error = "Error: More than one meeting booking ID provided."
    elif len(matches) == 1:
        meeting_booking_id = matches[0][:7]

    return meeting_booking_id, meeting_booking_error



import re

def extract_cab_booking_id(input_text):
    # Print input text for debugging
    print(f"DEBUG: Input text received: {input_text}")

    cab_booking_id_pattern = r'[Cc]\d{6,}'  # Match 'C' or 'c' followed by 6 or more digits
    matches = re.findall(cab_booking_id_pattern, input_text)

    # Debugging: Check what IDs are being found

    cab_booking_id = None
    cab_booking_error = None

    if len(matches) > 1:
        cab_booking_error = "Error: More than one cab booking ID provided."
    elif len(matches) == 1:
        # Only keep the first 6 characters following 'C'
        cab_booking_id = matches[0][:7]  # 'C' plus 6 digits

    return cab_booking_id, cab_booking_error








# Format the output with separated error messages and skip entities with errors
def format_output(intent, entities, errors):
    intent = intent.replace("intent: ", "").strip()

    output_parts = [f"intent= {intent}"]

    # Include entity only if there is no error associated with it
    if entities.get("DATE") and not errors.get("date_error"):
        output_parts.append(f"meeting_date= {entities['DATE']}")
    if entities.get("START_TIME") and not errors.get("time_error"):
        output_parts.append(f"starting_time= {entities['START_TIME']}")
    if entities.get("END_TIME") and not errors.get("time_error"):
        output_parts.append(f"ending_time= {entities['END_TIME']}")
    if entities.get("HALL_NAME") and not errors.get("hall_name_error"):
        output_parts.append(f"hall_name= {entities['HALL_NAME']}")
    if entities.get("NO_OF_PERSONS") and not errors.get("no_of_persons_error"):
        output_parts.append(f"no_of_persons= {entities['NO_OF_PERSONS']}")
    if entities.get("CAB_BOOKING_ID") and not errors.get("cab_booking_error"):
        output_parts.append(f"cab_booking_id= {entities['CAB_BOOKING_ID']}")
    if entities.get("MEETING_BOOKING_ID") and not errors.get("meeting_booking_error"):
        output_parts.append(f"meeting_booking_id= {entities['MEETING_BOOKING_ID']}")
    if entities.get("CAB_NAME") and not errors.get("cab_name_error"):
        output_parts.append(f"cab_name= {entities['CAB_NAME']}")
    if entities.get("BATCH_NO") and not errors.get("batch_no_error"):
        output_parts.append(f"batch_name= {entities['BATCH_NO']}")
    if entities.get("DROPPING_POINTS") and not errors.get("dropping_point_error"):
        output_parts.append(f"dropping_point= {entities['DROPPING_POINTS']}")

    # Separately mark each error
    if errors.get("date_error"):
        output_parts.append(f"error_date= {errors['date_error']}")
    if errors.get("hall_name_error"):
        output_parts.append(f"error_hall_name= {errors['hall_name_error']}")
    if errors.get("no_of_persons_error"):
        output_parts.append(f"error_no_of_persons= {errors['no_of_persons_error']}")
    if errors.get("time_error"):
        output_parts.append(f"error_time= {errors['time_error']}")
    if errors.get("batch_no_error"):
        output_parts.append(f"error_batch_no= {errors['batch_no_error']}")
    if errors.get("cab_name_error"):
        output_parts.append(f"error_cab_name= {errors['cab_name_error']}")
    if errors.get("cab_booking_error"):
        output_parts.append(f"error_cab_booking_id= {errors['cab_booking_error']}")
    if errors.get("meeting_booking_error"):
        output_parts.append(f"error_meeting_booking_id= {errors['meeting_booking_error']}")
    if errors.get("dropping_point_error"):
        output_parts.append(f"error_dropping_point= {errors['dropping_point_error']}")

    return ",".join(output_parts)



@app.post("/predict")
async def predict_intent(input_text: InputText):
    # Perform spell check and correction
    corrected_text = correct_spelling_with_languagetool(input_text.text)
    print("Loading into the model")
    # Extract entities
    dates = extract_dates(corrected_text)
    converted_dates = convert_dates(dates)
    meeting_date = converted_dates[0] if converted_dates else None
    #hall_name, hall_name_error = extract_hall_name(corrected_text)
    batch_no, batch_no_error = extract_batch_no(corrected_text)
    cab_name, cab_name_error = extract_cab_names(corrected_text)
    cab_booking_id, cab_booking_error = extract_cab_booking_id(corrected_text)
    meeting_booking_id, meeting_booking_error = extract_meeting_booking_id(corrected_text)
    no_of_persons, no_of_persons_error = extract_no_of_persons(corrected_text)
    start_time, end_time, time_error = extract_times(corrected_text)
    found_points, dropping_point_error = check_dropping_points(corrected_text)

    # Validate meeting date for being in the past
    meeting_date_error = validate_meeting_date(meeting_date) if meeting_date else None

    predicted_intent = predict(corrected_text)
    print(predicted_intent)

    if predicted_intent == "cab_booking":
        # Check if the starting time is either 18:30 or 19:30
        if start_time not in ["18:30", "19:30"]:
            # If the time doesn't match, set an error for invalid start time
            start_time = None  # Reset the start time to prevent using invalid time



    # Create an entity and error dictionary to pass into format_output
    entities = {
        "DATE": meeting_date,
        "START_TIME": start_time,
        "END_TIME": end_time,
        "HALL_NAME": None,
        "NO_OF_PERSONS": no_of_persons,
        "CAB_BOOKING_ID": cab_booking_id,
        "CAB_NAME": cab_name,
        "BATCH_NO": batch_no,
        "MEETING_BOOKING_ID": meeting_booking_id,
        "DROPPING_POINTS": found_points
    }

    errors = {
        "date_error": meeting_date_error,
        "time_error": time_error,
        "hall_name_error": None,
        "no_of_persons_error": no_of_persons_error,
        "cab_name_error": cab_name_error,
        "cab_booking_error": cab_booking_error,
        "batch_no_error": batch_no_error,
        "meeting_booking_error": meeting_booking_error,
        "dropping_point_error": dropping_point_error
    }

    # Format and return the output with separated errors
    output = format_output(predicted_intent, entities, errors)
    return {output}


# Meeting date validation to check if the date is in the past
def validate_meeting_date(meeting_date_str):
    current_date = datetime.now().strftime('%d/%m/%Y')
    meeting_date_obj = datetime.strptime(meeting_date_str, '%d/%m/%Y')
    current_date_obj = datetime.strptime(current_date, '%d/%m/%Y')

    if meeting_date_obj < current_date_obj:
        return "Please don't provide past dates."
    return None


# Start ngrok tunnel and get the public URL
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

# Run the FastAPI application
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
