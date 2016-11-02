import subprocess


def spawn_process(python_file, *args):
    """

    :param python_file:
    :param args:
    :return: (structured tuple)
    """
    try:
        completed_process = subprocess.run(['python', python_file].extend(args), check=True)
        # TODO: Use information from completed_process
    except subprocess.CalledProcessError as e:
        print(e.args)
        # TODO: handle error case
    except subprocess.TimeoutExpired as e:
        print(e.args)
        # TODO: handle error case

    # TODO: Explore IPC via stdin/stdout with subprocess.run
