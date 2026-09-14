import os, glob, json
print("cwd:", os.getcwd())
print("/kaggle exists:", os.path.isdir("/kaggle"))
if os.path.isdir("/kaggle/input"):
    print("input entries:", os.listdir("/kaggle/input"))
    for d in glob.glob("/kaggle/input/*"):
        try:
            print(d, "->", sorted(os.listdir(d))[:12])
        except Exception as e:
            print(d, "ERR", e)
else:
    print("NO /kaggle/input")
