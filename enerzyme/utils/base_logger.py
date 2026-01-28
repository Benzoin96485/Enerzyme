import logging, os, datetime, pathlib
from logging.handlers import TimedRotatingFileHandler
import enerzyme


BASE_NAME = "Enerzyme"
PACKAGE_PATH = str(pathlib.Path(enerzyme.__file__).parent.parent)

class PackagePathFilter(logging.Filter):
    def filter(self, record):
        """add relative path to record
        """
        pathname = record.pathname
        record.relativepath = None
        if pathname.startswith(PACKAGE_PATH):
            record.relativepath = os.path.relpath(pathname, PACKAGE_PATH)
        return True


class Logger(object):
    def __init__(self, logger_name='None'):
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)
        self.log_file_name = '{0}_{1}.log'.format(BASE_NAME, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        cwd_path = os.path.abspath(os.getcwd())
        self.log_path = os.path.join(cwd_path, "logs")

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.backup_count = 5

        self.console_output_level = 'INFO'
        self.file_output_level = 'INFO'
        self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter("%(asctime)s | %(relativepath)s:%(lineno)s | %(levelname)s | %(name)s | %(message)s", self.DATE_FORMAT)

    def get_logger(self):
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            console_handler.addFilter(PackagePathFilter())
            self.logger.addHandler(console_handler)

            file_handler = TimedRotatingFileHandler(filename=os.path.join(self.log_path, self.log_file_name), when='D',
                        interval=1, backupCount=self.backup_count, delay=True, encoding='utf-8')
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self.logger.addHandler(file_handler)
        return self.logger


logger = Logger(BASE_NAME).get_logger()
logger.setLevel(logging.INFO)