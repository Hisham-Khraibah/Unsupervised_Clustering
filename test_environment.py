import sys
import importlib

def get_required_packages():
    required_packages = {}
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                package_info = line.strip().split('==')
                if len(package_info) == 2:
                    package_name, package_version = package_info
                    required_packages[package_name] = package_version
    except Exception as e:
        print("Error reading requirements.txt:", e)
    return required_packages

def check_package_versions():
    try:
        required_packages = get_required_packages()
        for package_name, required_version in required_packages.items():
            try:
                # Define import name for each package
                if package_name == "scikit_learn":
                    import_name = "sklearn"
                else:
                    import_name = package_name.replace('-', '_')

                module = importlib.import_module(import_name)
                installed_version = getattr(module, '__version__', 'unknown')

                if installed_version != required_version:
                    print(f"Warning: {package_name} version {installed_version} is installed, but version {required_version} is required.")

            except ImportError:
                print(f"Error: {package_name} is not installed.")

            except Exception as e:
                print(f"Error checking {package_name}: {e}")

        print("Environment check completed.")

    except Exception as e:
        print("Error while checking environment:", e)

if __name__ == '__main__':
    try:
        check_package_versions()
        sys.exit(0)
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)