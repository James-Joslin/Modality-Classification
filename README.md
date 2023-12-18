# Modality-Classification
Classification models to determine treatment modality deployed into a compute cluster

### Summary
- API takes the form of a RESTful C# API
- Consists of two end points: GET and POST, for model download, and remote inference computation
- CORS enabled, and allows for Javascript based web application interaction
- Compatible with Pythonic, PHP, Javascript and C# backend requests
- API holds no data, and instead was built to be confined to a Kubernetes Pod and Query a remote database within another pod
- Hosted models are a pair of deep fully connected neural networks
- Prediction encompasses a 2-stage inference parse
- Evidence of application provided
### TODO
- Large scale batch processing
- GPU acceleration
- Model complexity expansion
