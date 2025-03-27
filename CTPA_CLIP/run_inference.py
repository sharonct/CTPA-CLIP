from ctclip_inference import CTClipInference
from pretrained_model import ctclip


inference: CTClipInference = CTClipInference(
    ctclip,
    data_folder = '/teamspace/studios/this_studio/data/train_preprocessed',
    reports_file= "/teamspace/studios/this_studio/data/train_reports.csv",
    labels = "/teamspace/studios/this_studio/data/inferred_labels.csv",
    batch_size = 1,
    results_folder="/teamspace/studios/this_studio/data/inference_zeroshot/",
    num_train_steps = 1,
)

inference.infer()