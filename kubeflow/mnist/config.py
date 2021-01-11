configmap = {}
with open("config.env") as f:
    for line in f:
        name, var = line.partition("=")[::2]
        configmap[name.strip()] = var.rstrip("\n")
