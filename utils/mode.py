import os


def get_py_files():
    res = []
    for root, dirs, files in os.walk("./"):
        if not any(substring in root for substring in ["venv", ".idea", "__pycache__"]):
            for file in files:
                if file.endswith(".py"):
                    res.append(os.path.join(root, file))
    return res


def mode_control(file_list: list, mode="release"):
    if mode == "debug":
        for file in file_list:
            inplace_change(file, "#@njit", "#@njit")
    if mode == "release":
        for file in file_list:
            inplace_change(file, "#@njit", "#@njit")


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)
