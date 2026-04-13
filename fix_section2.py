with open('add_section2.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Fix __getitem__ wrong path
code = code.replace(
    "img = self.tfm(Image.open(f\"{DATA_DIR}/{self.split_}/{s['filename']}.jpg\").convert(\"RGB\"))",
    "img = self.tfm(Image.open(f\"{DATA_DIR}/{self.split_name}/images/{s['filename']}.jpg\").convert(\"RGB\"))"
)

# 2. Remove split_ property
code = code.replace(
    "    @property\n    def split_(self): return \"train\"\n\nclass QuickDSSplit(QuickDS):\n    def __init__(self, split, n=128):\n        self.split_name = split\n        super().__init__(split, n)\n    @property\n    def split_(self): return self.split_name",
    "\nQuickDSSplit = QuickDS  # Alias"
)

with open('add_section2.py', 'w', encoding='utf-8') as f:
    f.write(code)

# Verify
with open('add_section2.py', 'r', encoding='utf-8') as f:
    code2 = f.read()
idx = code2.find('def __getitem__')
print("Fixed __getitem__:", repr(code2[idx:idx+200]))
print("QuickDSSplit alias present:", "QuickDSSplit = QuickDS" in code2)
print("split_ property gone:", "def split_" not in code2)
