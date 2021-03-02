
class BuildData:

    __datanames__ = ["train.txt", "dev.txt", "test.txt"]

    @staticmethod
    def create_finaldata(source:str, outpath:str):
        """
        Transform the data to the conll format.

        Args:
            source (str): The raw data path
            outpath (str): modified data path
        """
        assert outpath.split("/")[-1] not in BuildData.__datanames__,\
                f"Finalized data name must be one of these -> {BuildData.__datanames__}"

        with open(source, "r") as f:
            with open(outpath, "w") as fw:
                for line in f.readlines():
                    if line != "\n":
                        line = line.split("\t")
                        text = line[1].strip("\n")
                        entity = line[0]
                        fw.write(text + " " + entity + "\n")
                    else:
                        fw.write("\n")