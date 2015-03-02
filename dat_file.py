import pandas as pd
import re

class DatFile:
    def __init__(self, filename):
        self.filename = filename

        column = 1
        self.columns = []
        metadata = {}

        with open(filename, 'r') as f:
            for i in range(100):
                line = f.next().rstrip('\n\t\r')

                if line.startswith('# Column'):
                    column = int(self.find_number(line))
                    metadata[column] = {}
                elif line.startswith('#\tname'):
                    name = line.split(': ', 1)[1]
                    metadata[column]['name'] = name
                    self.columns.append(name)

            """
            for line in f:
                line = line.rstrip('\n\t\r')

                if line.startswith('# Column'):
                    column = int(self.find_number(line))
                    metadata[column] = {}

                elif line.startswith('#\tstart'):
                    metadata[column]['start'] = float(self.find_number(line))

                elif line.startswith('#\tend'):
                    metadata[column]['end'] = float(self.find_number(line))

                elif line.startswith('#\tsize'):
                    metadata[column]['size'] = int(self.find_number(line))

                elif line.startswith('#\tname'):
                    name = line.split(': ', 1)[1]
                    metadata[column]['name'] = name
                    self.columns.append(name)
            """

        self.meta = {}
        for key in metadata:
            self.meta[self.columns[key - 1]] = metadata[key]

        self.df = pd.read_table(filename, engine='c', sep='\t', comment='#', names=self.columns)

    def find_number(self, s):
        return re.findall('[-+]?\d*\.\d+|\d+', s)[0] or None