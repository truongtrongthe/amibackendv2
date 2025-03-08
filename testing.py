import fasttext
model = fasttext.load_model("lid.176.bin")
print(model.predict("Chào em", k=1))  # Check the output