from fastapi import FastAPI
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

class EMR(BaseModel):
    patient_info: dict
    visit_info: dict
    chief_complaint: str
    present_illness: str
    past_history: str
    vital_signs: dict
    lab_results: str
    imaging_results: str
    diagnosis: str
    plan: str
    free_text: str

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = PreTrainedTokenizerFast.from_pretrained("emr_summary_kobart")
model = BartForConditionalGeneration.from_pretrained("emr_summary_kobart").to(device)

def convert_to_prompt(emr):
    return f"""
[환자정보] {emr.patient_info.get('age')}세 {emr.patient_info.get('sex')}
[주호소] {emr.chief_complaint}
[현병력] {emr.present_illness}
[과거력] {emr.past_history}
[활력징후] {emr.vital_signs}
[검사] {emr.lab_results}
[영상] {emr.imaging_results}
[진단] {emr.diagnosis}
[계획] {emr.plan}
[기타] {emr.free_text}
""".strip()

@app.post("/summarize")
def summarize(emr: EMR):
    text = convert_to_prompt(emr)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        num_beams=5
    )

    summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "summary": {
            "full_summary_text": summary_text
        }
    }
