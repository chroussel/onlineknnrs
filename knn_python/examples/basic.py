import knn_py

service = knn_py.KnnService()
model_name = "tf"
service.load_country("FR", "../data/indices", "../data/embeddings")
service.load_model("FR", model_name, "../data/models/country=FR/_model.pb")
knn_result = service.query("FR", 782, 10, [(782, 439154173303199114, 1580637528, 2)])
print(knn_result)
knn_tf_result = service.tf_query("FR", 782, 10, [(782, 439154173303199114, 1580637528, 2)], model_name)
print(knn_tf_result)