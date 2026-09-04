import sys
import os
import tempfile
import time

try:
    from fastapi import FastAPI, Request
    import uvicorn
except ImportError:
    print("Error: fastapi and uvicorn are required. Run: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="LiteClaw Nemotron ASR Server")

print("Initializing ASR Server...")
model = None

def load_model():
    global model
    try:
        import nemo.collections.asr as nemo_asr
        model_name = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
        print(f"Loading NeMo model: {model_name}...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load NeMo model: {e}")
        print("Please ensure nemo_toolkit[asr] and torch are installed.")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/transcribe")
async def transcribe(request: Request):
    if model is None:
        return {"text": "", "language": "en", "error": "ASR model is not loaded"}

    # Read binary WAV from the request body
    body = await request.body()
    if not body:
        return {"text": "", "language": "en", "duration_ms": 0}

    # Write WAV to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.write(body)
        temp_wav_path = temp_wav.name

    start_time = time.time()
    try:
        # NeMo transcribe expects file paths
        transcriptions = model.transcribe([temp_wav_path])
        duration_ms = int((time.time() - start_time) * 1000)

        text = ""
        if transcriptions:
            res = transcriptions[0]
            if isinstance(res, tuple):
                text = res[0]
            else:
                text = res

        return {
            "text": text,
            "language": "en",
            "duration_ms": duration_ms
        }
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"text": "", "language": "en", "error": str(e), "duration_ms": int((time.time() - start_time) * 1000)}
    finally:
        try:
            os.remove(temp_wav_path)
        except Exception:
            pass

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089)
