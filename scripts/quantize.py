from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType


model_id = "jdirkx/xlmroberta-enja-ner"

model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
model.save_pretrained("onnx_model")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("onnx_model")

quantize_dynamic(
    model_input="onnx_model/model.onnx",
    model_output="onnx_model/model-quant.onnx",
    weight_type=QuantType.QInt8
)