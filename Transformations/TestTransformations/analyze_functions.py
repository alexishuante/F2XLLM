import clang.cindex

def analyze_function(node):
    """Analyzes any function declaration node from the AST."""
    signature = {
        "name": node.spelling,
        "return_type": node.result_type.spelling,
        "parameters": [],
        "unrecognized_keywords": []  # Track unrecognized keywords
    }

    known_keywords = set(["const", "volatile", "restrict", "static", "extern", "register", "inline", "_Atomic",
                        "signed", "unsigned", "short", "long", "char", "int", "float", "double", "void"])
    # Now known_keywords is a set, making the 'word not in known_keywords' check more efficient.

    for arg in node.get_arguments():
        param = {
            "name": arg.spelling,
            "type": arg.type.spelling,
            "is_pointer": arg.type.kind == clang.cindex.TypeKind.POINTER
        }

        # Keyword analysis
        words = arg.type.spelling.split()
        for word in words:
            if word not in known_keywords:
                signature["unrecognized_keywords"].append(word)

        signature["parameters"].append(param)

    return signature

# Ensure libclang is accessible (adjust the path if necessary)
clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang.so.10')  # Example path for libclang
index = clang.cindex.Index.create()

# Replace with the path to your C file containing the functions
file_path = "your_c_file.c"  
tu = index.parse(file_path)

for node in tu.cursor.walk_preorder():
    if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        signature = analyze_function(node)
        if signature["unrecognized_keywords"]:
            print(f"Unrecognized keywords in function '{node.spelling}': {signature['unrecognized_keywords']}")
        print(signature)  # Print the signature
