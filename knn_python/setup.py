from setuptools import setup, find_packages
from setuptools_rust import RustExtension, Binding
from setuptools_rust import build_ext as _build_ext
import os
import shutil
import subprocess
import platform

def get_tf_os():
    plat = platform.system().lower()
    if plat == "darwin":
        return "darwin"
    elif plat == "linux":
        return "linux"
    elif plat == "windows":
        return "windows"
    else:
        print("Unsupported platform " + plat)
        exit(1)

TF_DIR = os.path.abspath("build/external/tensorflow")
TF_VERSION = "1.13.1"

TF_OS = get_tf_os()

class knn_build(_build_ext):
    def download_tf_library(self):
        print("Download TF to " + TF_DIR)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print(os.path.relpath(__file__))
        download_script = os.path.join(script_dir, "download_and_install_tf.sh")
        print("executing "+ download_script)
        command = [download_script, TF_OS, TF_VERSION, TF_DIR]
        result = subprocess.check_output(command)
        print(result)

    def copy_tf_library(self):
        target_libpath = os.path.dirname(_build_ext.get_ext_fullpath(self, "libtensorflow"))
        for (root, dirs, files) in os.walk(TF_DIR):
            if os.path.basename(root) == "lib":
                for library in files:
                    shutil.copyfile(os.path.join(root, library), os.path.join(target_libpath, library))

    def check_extensions_list(self, extensions):
        if extensions:
            _build_ext.check_extensions_list(self, extensions)

    def run(self):
        super().run()
        print("running knn_build")
        if not os.path.exists(TF_DIR):
            self.download_tf_library()
        self.copy_tf_library()



setup_requires = ["setuptools-rust>=0.10.1", "wheel"]
install_requires = []

setup(
    name="knn_python",
    version="0.1.0",
    classifiers=["Development Status :: 4 - Beta",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved :: Apache Software License",
                 "License :: OSI Approved",
                 "Operating System :: MacOS",
                 "Operating System :: Microsoft :: Windows",
                 "Operating System :: POSIX :: Linux",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python",
                 "Programming Language :: Rust"
                 ],
    rust_extensions=[RustExtension("knn_python", quiet=False, binding=Binding.PyO3, debug=False)],
    install_requires=install_requires,
    setup_requires=setup_requires,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    cmdclass=dict(build_ext=knn_build)

)
