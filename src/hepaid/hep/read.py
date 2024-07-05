"""Collection of utility function and classes to read SLHA files, HiggsBounds/HiggsSignals output and MadGraph results.

Key Features:
- Utilities for reading and writing SLHA blocks, supporting both generals BLOCK and DECAY types.
- Parsing and manipulation of SLHA files, including block extraction and conversion to dictionaries.
- Tools for handling HiggsBounds and HiggsSignals results.
- Utilty class to create example MadGraph scripts.

Example:
    >>> from hep_aid import SLHA
    >>> slha_file = SLHA("path_to_slha_file")
"""
import re
import os
from shutil import copy
import numpy as np
from pathlib import Path

from collections.abc import MutableMapping, Mapping
from typing import Dict, List, Tuple, Union
import warnings



PATTERNS_SLHA = dict(
    block_header=r"(?P<block>BLOCK)\s+(?P<block_name>\w+)(\s+)?((Q=.*)?(?P<q_values>-?\d+\.\d+E.\d+))?(\s+)?(?P<comment>#.*)?",
    nmatrix_value=r"(?P<entries>.+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)",
    model_param_pattern=r"(?P<entries>.+)\s+(?P<comment>#.*)",
    decay_header=r"DECAY\s+(?P<particle>\w+)\s+(?P<value>-?\d+\.\d+E.\d+)(\s+)?(?P<comment>#.*)?",
    decay1l_header=r"DECAY1L\s+(?P<particle>\w+)\s+(?P<value>-?\d+\.\d+E.\d+)(\s+)?(?P<comment>#.*)",
    decay_body_pattern=r"(?P<value>.?\d+\.\d+E.\d+)\s+(?P<entries>.+)\s+(?P<comment>#.*)",
)


SPHENO_ON_OFF_BLOCKS = ['SPHENOINPUT', 'DECAYOPTIONS', 'MODSEL' ]

def find_block(name: str, block_list: list):
    """Finds a block in a list of blocks by its name.

    Parameters:
        name: The name of the block to search for.
        block_list: A list of strings with Blocks Names.

    Returns:
        The first BlockSLHA object with a matching name.

    Raises:
        ValueError: If the block name is not found.
    """
    if isinstance(name, str):
        for b in block_list:
            b_found = b if b.block_name == name else None
            if b_found is not None:
                break
        if b_found is None:
            raise ValueError("Block not found with the specified block name.")
        return b_found

def on_off_line_formatter(
        entries,
        value,
        comment,
        long: bool = True
        ) -> str:
    if long:
        entries_formatted = f"{entries[0]}".rjust(12)
    else:
        entries_formatted = f"{entries[0]}".rjust(4)

    value_formatted = f"{value[0]}".rjust(12) 

    spaces = "  "
    formatted_line = ""
    formatted_line += f"{entries_formatted}"
    formatted_line += f"{value_formatted}"
    formatted_line += f"{spaces}{comment}"
    formatted_line += '\n'
    return formatted_line

def decay_line_formatter(
        entries,
        value,
        comment,
        long: bool = True
        ) -> str:
    """Formats decay block line elements from `extract_line_elements` into a standardized 
    decay string `"3.08079654E-02           2          -4           4    # BR(HH_1 -> FU_2^* FU_2 ))"`.

    Parameters:
        entries (tuple or list): Decay parameters or indices (integers).
        value (float): Value to format in scientific notation.
        comment (str): Descriptive comment.
        long: bool = Default False. Determines the spacing between entries.

    Returns:
        str: Formatted string with value in scientific notation, fixed-width
             entries, and the comment.
    """
    if is_comment_line(value, entries):
        formatted_line = comment
    else:

        spaces = "    "

        if len(value) == 1:
            value_formatted = f"{value[0]:.8E}".rjust(19)
        elif len(value) > 1:
            value_formatted = [f"{x:.8E}".rjust(19) for x in value]
        else:
            value_formatted = f""

        if long:
            entries_formatted = [f"{x}".rjust(12) for x in entries]
        else:
            entries_formatted = [f"{x}".rjust(4) for x in entries]

        formatted_line = f"{spaces}{value_formatted}"
        for i in range(len(entries)):
            formatted_line += f"{entries_formatted[i]}"
        formatted_line += f"{spaces}{comment}"
    formatted_line += '\n'
    return formatted_line

def is_on_off_line(entries, values):
    """ Define if a line is a ON/OFF type from SPheno input file.
    """
    # Wild conditional
    two_entries = len(entries) == 2 
    no_values = len(values) == 0
    return True if two_entries and no_values else False

def is_higgs_couplings_line(comment):
    return True if 'COUPLING' in comment else False
        
def is_comment_line(values: list, entries: list):
    """Utility function to determine whether a line in an SLHA block is solely a comment.

    Parameters:
        values (list): The list of numerical values associated with the line.
        entries (list): The list of entries (identifiers) at the beginning of the line.

    Returns:
        bool: True if the line is considered a comment line (empty values and entries), False otherwise.
    """
    no_values = len(values) == 0
    no_entries = len(entries) == 0
    if no_values and no_entries:
        return True
    else:
        return False

def nmatrix_line_formatter(
        entries,
        value,
        comment,
        long: bool = True 
        ) -> str:
    """Formats nmatrix block line elements from `extract_line_elements` into a standardized 
    generic nmatrix block string, `"1000039     3.28786550E+03   # CHI_7"`

    Parameters:
        entries (tuple or list): Decay parameters or indices (integers).
        value (float): Value to format in scientific notation.
        comment (str): Descriptive comment.
        long: bool = Default False. Determines the spacing between entries.

    Returns:
        str: Formatted string with value in scientific notation, fixed-width
             entries, and the comment.
    """

    if is_comment_line(value, entries):
        formatted_line = comment

    elif is_higgs_couplings_line(comment):
        spaces = "   "
        entries_formatted = [f"{x}".rjust(12) for x in entries]
        value_formatted = [f"{x:.8E}".rjust(19) for x in value]

        formatted_line = " "
        for i in range(len(value)):
            formatted_line += f"{value_formatted[i]}"
        for i in range(len(entries)):
            formatted_line += f"{entries_formatted[i]}"
        formatted_line += f"{spaces}{comment}"
    else:
        spaces = "   "
        if long:
            entries_formatted = [f"{x}".rjust(12) for x in entries]
        else:
            entries_formatted = [f"{x}".rjust(4) for x in entries]

        if len(value) == 1:
            value_formatted = f"{value[0]:.8E}".rjust(19)
        elif len(value) > 1:
            value_formatted = [f"{x:.8E}".rjust(19) for x in value]
        else:
            value_formatted = f""

        formatted_line = " "
        for i in range(len(entries)):
            formatted_line += f"{entries_formatted[i]}"
        formatted_line += f"{value_formatted}"
        formatted_line += f"{spaces}{comment}"
    formatted_line += '\n'
    return formatted_line

def extract_line_elements(line: str ) -> dict[str,list]:
    """
    Extracts various elements from a given line of text based on predefined patterns.

    Parameters:
    line (str): A string line from which elements are to be extracted.

    Returns:
    dict: A dictionary containing extracted elements categorized by their type:
          - 'comment': A string containing the comment portion of the line.
          - 'value': A list of floating-point numbers in scientific notation found in the line.
          - 'entries': A list of integer values found in the line.

    Example:
    Input: "123 1.23E10 # Example"
    Output: {'comment': ['# Example'], 'value': ['1.23E10'], 'entries': ['123'], 'line_category': 'STANDARD'}
    """

    patterns = dict(
        comment=r"(?P<comment>#.*)",
        values=r"(?P<value>.?\d+\.\d+E.\d+)",
        entries=r"(?P<entries>[+-]?\d+)",
    )
    line_elements = {}
    _line = line.upper()
    for p in patterns:
        line_elements[p] = re.findall(patterns[p], _line)
        _line = re.sub(patterns[p], "", _line)
       
    if is_on_off_line(line_elements['entries'], line_elements['values']):
        # If is on/off line from SPheno, the last entry is the value
        line_category = 'ONOFF'
        line_elements['values'].append(line_elements['entries'].pop())
    else:
        line_category = 'STANDARD'

    line_elements['line_category'] = line_category
    return line_elements


def block2dict(block):
    """
    Converts a given data block into a dictionary format, capturing detailed elements and
    properties of the block.

    Parameters:
    block (Block): An Block object with methods to access its elements
                   and properties such as entries, values, comments, and other metadata.

    Returns:
    dict: A dictionary containing all relevant data extracted from the block, structured as follows:
        - 'entries': A dictionary of entries where each key is a comma-joined string of entries,
                     and each value is another dictionary with 'value', 'comment', and 'line'.
        - 'block_name': The name of the block.
        - 'block_comment': The comment associated with the block.
        - 'q_values': The Q values associated with the block, if any.
        - 'block_category': The category of the block.
        - 'header_line': The header line of the block.
        - 'pid': (optional) The PID, if the block category is related to decay processes.
        - 'decay_width': (optional) The decay width, if applicable.

    """
    block_dict = {}
    entries_dict = {}
    for i, entries in enumerate(block.keys()):
        entries_dict[",".join(entries)] = {
            "value": block.values()[i],
            "comment": block.comments()[i],
            "line": block.lines()[i],
            'line_category': block.line_categories()[i]
        }
    block_dict["entries"] = entries_dict
    block_dict["block_name"] = block.block_name
    block_dict["block_comment"] = block.block_comment
    block_dict["q_values"] = block.q_values
    block_dict["block_category"] = block.block_category
    block_dict["header_line"] = block.header_line
    if (block.block_category == "DECAY") or (block.block_category == "DECAY1L"):
        block_dict["pid"] = block.pid
        block_dict["decay_width"] = block.decay_width
    return block_dict


def slha2dict(slha):
    """
    Converts the whole SLHA object into a dictionary.
    """
    slha_dict = {}
    for i, block in enumerate(slha.block_list):
        slha_dict[block] = block2dict(slha[block])
    return slha_dict



class BlockLineSLHA:
    """Represents a single line within an SLHA (Supersymmetric Les Houches Accord) file.
    Is built to be used with the   `extract_line_elements` function.

    Attributes:
        entries (list): The numerical or string identifiers at the start of the line.
        value (list): Numerical values associated with the line's entries.
        comment (str): Optional comment following the entries and values.
        line_category (str): Category of the line ("DECAY" or "BLOCK").
        line (str): The original, unmodified text of the line from the SLHA file for debugging.

    Methods:
        __init__(self, entries, value, comment, line_category, line=None): 
            Initialises a BlockLineSLHA object.
        __repr__(self): Returns a formatted string representation of the line,
                        adhering to SLHA formatting rules for decay or nmatrix lines.

    Example:
        line_data = BlockLineSLHA([1000022], [1.275], ["# Neutralino 1 mass"])
        print(line_data)  # Output:  1000022   1.27500000E+00  # Neutralino 1 mass
    """ 


    def __init__(self, 
                 entries: list, 
                 value: list, 
                 comment: list, 
                 line_category: str, 
                 block_category: str,
                 line: str = None
                 ):
        """Initialises a BlockLineSLHA object.

        Parameters:
            entries (list): The numerical or string identifiers at the start of the line.
            value (list): Numerical values associated with the line's entries.
            comment (str): Optional comment following the entries and values.
            line_category (str): Category of the line ("DECAY" or "BLOCK").
            line (str): The original, unmodified text of the line from the SLHA file for debugging.
        """
        self.entries = entries


        self.value = [float(v) if v != None else None for v in value]
        if line_category == 'ONOFF':
            self.value = [int(v) if v != None else None for v in value]

        self.comment = comment[0] if len(comment) == 1 else comment
        self._total_entries_list = [entries] + [value] + [comment]
        self.block_category = block_category
        self.line_category = line_category
        self.line = line


    def __repr__(self):
        """Returns a formatted string representation of the line.

        The formatting is specific to the line category ("DECAY" or "nmatrix" formats) and is
        defined by the functions `decay_line_formatter` and `nmatrix_line_formatter`.

        Returns:
            str: The formatted line string.
        """
        if self.block_category == "DECAY":
            line = decay_line_formatter(self.entries, self.value, self.comment)
        else:
            if self.line_category == 'ONOFF':
                line = on_off_line_formatter(self.entries, self.value, self.comment)
            else:
                line = nmatrix_line_formatter(self.entries, self.value, self.comment)
        return line
    


class BlockSLHA(MutableMapping):
    """Represents a complete block within an SLHA (Supersymmetric Les Houches Accord) file.
    This class inherits from `MutableMapping`, providing dict-like behavior. An SLHA Block 
    is essentially a dictionary that can be queried by the entries. The entries are the keys of 
    the dictionary. 

    Attributes:
        block_name (str): Name of the block (e.g., "MASS", "DECAY 25", "MODSEL").
        block_comment (str, optional): Optional comment describing the block.
        q_values (list, optional): Q values associated with the block (if applicable).
        block_body (list): List of BlockLineSLHA objects representing lines in the block.
        block_category (str, optional): Category of the block ("BLOCK" or "DECAY").
        header_line (str, optional): The first line of the block, often containing metadata.
        pid (int, optional): Particle ID (relevant for "DECAY" blocks).
        decay_width (float, optional): Decay width (relevant for "DECAY" blocks).

    Methods:
        __getitem__(self, key): Retrieves a BlockLineSLHA object based on its entries.
        __setitem__(self, key, value): Modifies the value of a BlockLineSLHA object.
        __iter__(self): Returns iterator over BlockLineSLHA objects in the block.
        __len__(self): Returns the number of lines in the block.
        __repr__(self): Returns a formatted string representation of the entire block.
        keys(self): Returns a list of entry lists from each BlockLineSLHA.
        values(self): Returns a list of value lists from each BlockLineSLHA.
        comments(self): Returns a list of comments from each BlockLineSLHA.
        lines(self): Returns a list of the original line strings from each BlockLineSLHA.
    """

    def __init__(
        self,
        block_name,
        block_comment=None,
        q_values=None,
        block_category=None,
        decay_width=None,
        header_line=None,
    ):
        """Initialises a new BlockSLHA object.

        Attributes:
        block_name (str): Name of the block (e.g., "MASS", "DECAY 25", "MODSEL").
        block_comment (str, optional): Optional comment describing the block.
        q_values (list, optional): Q values associated with the block (if applicable).
        block_body (list): List of BlockLineSLHA objects representing lines in the block.
        block_category (str, optional): Category of the block ("BLOCK" or "DECAY").
        header_line (str, optional): The first line of the block, often containing metadata.
        pid (int, optional): Particle ID (relevant for "DECAY" blocks).
        decay_width (float, optional): Decay width (relevant for "DECAY" blocks).
        """
        self.block_name = block_name
        self.block_comment = block_comment
        self.q_values = q_values
        self.block_body = []
        self.block_category = block_category
        self.header_line = header_line
        if (self.block_category == "DECAY") or (self.block_category == "DECAY1L"):
            self.pid = int(self.block_name.split()[-1])
            self.decay_width = float(decay_width)

    def __getitem__(self, key):
        """Retrieves a BlockLineSLHA object based on its entries.
        """
        return self._get(key)

    def __setitem__(self, key, value):
        """Modifies the value of a BlockLineSLHA object.
        """
        self._set(key, value)

    def __delitem__(self, key):
        pass

    def __iter__(self):
        """Returns iterator over BlockLineSLHA objects in the block.
        """
        return iter(self.block_body)

    def __len__(self):
        """Returns the number of lines in the block.
        """
        return len(self.block_body)

    def keys(self):
        """Returns a list of entry lists from each BlockLineSLHA.
        """
        entries = [i.entries for i in self]
        return entries

    def values(self):
        """Returns a list of value lists from each BlockLineSLHA.
        """
        values = [i.value for i in self]
        return values

    def comments(self):
        """Returns a list of comments lists from each BlockLineSLHA.
        """
        comments = [i.comment for i in self]
        return comments

    def lines(self):
        """Returns a list of unmodified line lists from each BlockLineSLHA.
        """
        lines = [i.line for i in self]
        return lines

    def line_categories(self):
        """Returns a list of unmodified line lists from each BlockLineSLHA.
        """
        lines = [i.line_category for i in self]
        return lines

    def __repr__(self):
        """Returns a formatted string representation of the entire block.
        """
        block_format = self.header_line
        for l in self.block_body:
            block_format += repr(l)
        return block_format

    def _set(self, find: Tuple[int], value: float | list) -> None:
        """Modifies the value of a line within the block.

        Searches for the line based on its entries (default) or comment, and then sets the 
        corresponding value  to the provided new value.

        Parameters:
            find (Tuple[int]): A tuple representing the numerical entries to search for.
            value (float or list): The new value to set (float for single values, list for multiple).

        Raises:
            ValueError: If the specified entries or comment are not found in the block.
        """

        # Define find iterable
        find = [find] if isinstance(find, int) else find
        find = [str(i) for i in find]

        for i, b in enumerate(self.block_body):
            line_entries = b.entries
            b_found = i if b.entries == find else None
            if b_found is not None:
                break

        if b_found is None:
                raise ValueError("Entry not found with the specified entries or comment.")

        output = self.block_body[i].value
        if isinstance(output, list):
            if len(output) == 1:
                self.block_body[i].value[0] = value
            else:
                self.block_body[i].value = value

    def _get(self, find: Tuple[Union[int, str]], request: str = "value") -> float:
        """Retrieves the value or comment of a line within the block.

        Searches for the line based on its entries (default) or comment, and then returns the corresponding value 
        or comment.

        Parameters:
            find (Tuple[int] or Tuple[str]): A tuple representing the numerical entries or a tuple of 
                                             strings representing the comment to search for.
            request (str, optional): Specifies whether to retrieve the "value" (default) or "comment" of the line.

        Returns:
            float or str: The value or comment associated with the first matching line.

        Raises:
            ValueError: If the specified entries or comment are not found in the block, or if the 
                        'request' argument is invalid.
        """

        # Define find iterable
        find = [find] if not isinstance(find, Tuple) else find
        find = [str(i) for i in find]

        # Define search according to value or comment
        if request == "value":
            get_ = lambda line: line.value
        if request == "comment":
            get_ = lambda line: line.comment

        for i, line in enumerate(self.block_body):
            b_found =  line if line.entries == find else None
            if b_found is not None:
                break
        if b_found is None:
                raise ValueError("Entry not found with the specified entries or comment.")

        output = get_(b_found)
        if isinstance(output, list):
            return output[0] if len(output) == 1 else output

def read_blocks_from_dict(slha_dict):
    """Reads SLHA blocks from a dictionary representation.

    This function takes a dictionary where keys represent SLHA block names and values are dictionaries 
    containing block data. 

    Parameters:
        slha_dict: A dictionary containing SLHA data.

    Returns:
        A list of `BlockSLHA` objects representing the parsed blocks. Each `BlockSLHA` object
        stores information about the block's type, name, comment, values, and body (if applicable).
    """

    block_list = []
    for i, block in enumerate(slha_dict.keys()):
        if slha_dict[block]["block_category"] == "BLOCK":
            block_list.append(
                BlockSLHA(
                    block_name=slha_dict[block]["block_name"],
                    block_comment=slha_dict[block]["block_comment"],
                    q_values=slha_dict[block]["q_values"],
                    block_category=slha_dict[block]["block_category"],
                    header_line=slha_dict[block]["header_line"],
                )
            )
        else:
            block_list.append(
                BlockSLHA(
                    block_name=slha_dict[block]["block_name"],
                    block_comment=slha_dict[block]["block_comment"],
                    q_values=slha_dict[block]["q_values"],
                    block_category=slha_dict[block]["block_category"],
                    header_line=slha_dict[block]["header_line"],
                    decay_width=slha_dict[block]["decay_width"],
                )
            )
        for entry in slha_dict[block]["entries"].keys():
            block_list[-1].block_body.append(
                BlockLineSLHA(
                    entries=entry.split(","),
                    value=slha_dict[block]["entries"][entry]["value"],
                    comment=slha_dict[block]["entries"][entry]["comment"],
                    block_category=slha_dict[block]["block_name"].split()[0],
                    line_category=slha_dict[block]["entries"][entry]['line_category'],
                    line=slha_dict[block]["entries"][entry]["line"],
                )
            )
    return block_list

def read_blocks_from_file(file_dir: str) -> list:
    """Reads blocks of SLHA (SUSY Les Houches Accord) data from a file.

    This function parses an SLHA file, identifying and extracting blocks of data 
    defined by specific headers. It supports SLHA block types: BLOCK, DECAY, and DECAY1L. 
    The function skips over lines containing "SPINFO".

    Parameters:
        file_dir (str): The path to the SLHA file.

    Returns:
        A list of `BlockSLHA` objects, each representing a parsed block of SLHA data.
        The `BlockSLHA` object stores information about the block's type, name, comment,
        values, and body (if applicable).
    """
    block_list = []
    in_block = None
    with open(file_dir, "r") as file:
        for line in file:
            if "SPINFO" in line.upper().strip():
            # Implement functionality to properly read this block. 
                continue

            m_block = re.match(PATTERNS_SLHA["block_header"], line.upper().strip())
            if not (m_block == None):
                block_list.append(
                    BlockSLHA(
                        block_name=m_block.group("block_name"),
                        block_comment=m_block.group("comment"),
                        q_values=m_block.group("q_values"),
                        block_category="BLOCK",
                        header_line=line,
                    )
                )
                in_block, block_from = m_block.group("block_name"), "parameter_data"
                continue

            m_block = re.match(PATTERNS_SLHA["decay_header"], line.upper().strip())
            if not (m_block == None):
                block_name = "DECAY {}".format(m_block.group("particle"))
                block_list.append(
                    BlockSLHA(
                        block_name=block_name,
                        block_comment=m_block.group("comment"),
                        block_category="DECAY",
                        decay_width=m_block.group("value"),
                        header_line=line,
                    )
                )
                in_block, block_from = block_name, "decay_data"
                continue

            m_block = re.match(PATTERNS_SLHA["decay1l_header"], line.upper().strip())
            if not (m_block == None):
                block_name = "DECAY1L {}".format(m_block.group("particle"))
                block_list.append(
                    BlockSLHA(
                        block_name=block_name,
                        block_comment=m_block.group("comment"),
                        block_category="DECAY1L",
                        decay_width=m_block.group("value"),
                        header_line=line,
                    )
                )
                in_block, block_from = block_name, "decay_data"
                continue

            if in_block is not None:
                line_elements = extract_line_elements(line)
                find_block(in_block, block_list).block_body.append(
                    BlockLineSLHA(
                        entries=line_elements["entries"],
                        value=line_elements["values"],
                        comment=line_elements["comment"],
                        block_category=in_block.split()[0],
                        line_category=line_elements['line_category'],
                        line=line,
                    )
                )
        return block_list

class SLHA(Mapping):
    """Reads and manages SLHA (SUSY Les Houches Accord) files.

    This class provides a convenient way to parse, access, and manipulate data within 
    the file. Stores each block of an SLHA file as BlockSLHA objects, each BlockSLHA
    contains a list of BlockLineSLHA to extract entries and values.
    This class inherits from `Mapping`, providing dict-like behavior.

    Attributes:
        _blocks (list): A list of BlockSLHA objects, each representing a block in the SLHA file.
        block_list (list): A list of block names in the SLHA file.

    Methods:
        __init__(file): Initializes an SLHA object from a file path or dictionary.
        __getitem__(key): Retrieves a BlockSLHA object by name.
        __repr__(): Returns a string representation of the SLHA object.
        __iter__(): Returns an iterator over the BlockSLHA objects.
        __len__(): Returns the number of blocks in the SLHA file.
        keys(): Returns a list of block names.
        block(name): Retrieves a BlockSLHA object by name.
        as_txt(): Returns the SLHA file content as a string.
        as_dict(): Returns the SLHA file content as a dictionary.
        save(path): Saves the SLHA data to a new file.

    Example:
        ```python
        slha = SLHA("param_card.dat")
        block = slha["SMINPUTS"]
        value = block["1"]   # Access a value within the block
        print(slha.as_dict()) 
        slha.save("modified_param_card.dat")
        ```

    Parameters:
        file (str | dict): The path to the SLHA file or a dictionary containing the data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the input file is not a valid SLHA file or if the dictionary is malformed.
    """


    def __init__(
        self,
        file: str | dict,
    ) -> None:
        """Initialise an SLHA instance

        Parameters:
            file (str | dict): The path to the SLHA file or a SLHA dictionary containing the data.
        """

        # Initialize
        if isinstance(file, dict):
            self._blocks = read_blocks_from_dict(file)
        else:
            self._blocks = read_blocks_from_file(file)
        self.block_list = [name.block_name for name in self._blocks]

    def __getitem__(self, key):
        """Retrieves a Block from the SLHA file.
        """
        return self.block(key)

    def __repr__(self):
        return "SLHA. {} blocks".format(len(self.block_list))

    def __iter__(self):
        """Returns iterator over the Blocks.
        """
        return iter(self._blocks)

    def __len__(self):
        """Return the number of blocks in the SLHA file.
        """
        return len(self.block_list)

    def keys(self):
        """Returns a list with the names of all the SLHA Blocks.
        """
        return self.block_list

    def block(self, name):
        block = find_block(name.upper(), self._blocks)
        return block

    def as_txt(self):
        """ Return blocks as texts
        """
        txt = ""
        for b in self._blocks:
            txt += b.__repr__()
        return txt

    def as_dict(self):
        """Return SLHA data as a dictionary"""
        return slha2dict(self)

    def save(self, path: str):
        """Write the SLHA instance in a new file in `path`.

        Parameters:
            path (str): path/to/the/new_file
        """
        dir_path = Path(path)
        dir_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dir_path, "w+") as f:
            for block in self._blocks:
                f.write(str(block))

    


class MG5Script:
    """
    Create a object containing mostly* all the necesary commands to compute a process in madgraph with the \n
    text-file-input/script mode. A default or template input file can be preview with the show() method within this class.
    """

    def __init__(self, work_dir, ufo_model):
        self.work_dir = work_dir
        self.ufo_model = ufo_model
        self._default_input_file()

    def import_model(self):
        out = "import model {} --modelname".format(self.ufo_model)
        self._import_model = out

    def define_multiparticles(
        self,
        syntax=[
            "p = g d1 d2  u1 u2  d1bar d2bar u1bar u2bar",
            "l+ = e1bar e2bar",
            "l- = e1 e2",
        ],
    ):
        out = []
        if not (syntax == None):
            [out.append("define {}".format(i)) for i in syntax]
            self._define_multiparticles = out
        else:
            self._define_multiparticles = None

    def process(self, syntax="p p > h1 > a a"):
        """
        Example: InputFile.process('p p > h1 > a a')
        """
        out = "generate {}".format(syntax)
        self._process = out

    def add_process(self, syntax=None):
        """
        Example: InputFile.add_process('p p > h2 > a a')
        """
        out = []
        if not (syntax == None):
            [out.append("add process {}".format(i)) for i in syntax]
            self._add_process = out
        else:
            self._add_process = None

    def output(self, name="pph1aa"):
        output_dir = os.path.join(self.work_dir, name)
        out = "output {}".format(output_dir)
        self._output = out

    def launch(self, name="pph1aa"):
        launch_dir = os.path.join(self.work_dir, name)
        out = "launch {}".format(launch_dir)
        self._launch = out

    def shower(self, shower="Pythia8"):
        """
        Call .shower('OFF') to deactivate shower effects.
        """
        out = "shower={}".format(shower)
        self._shower = out

    def detector(self, detector="Delphes"):
        """
        Call .detector('OFF') to deactivate detector effects.
        """
        out = "detector={}".format(detector)
        self._detector = out

    def param_card(self, path=None):
        if path == None:
            self._param_card = None
        else:
            self._param_card = path

    def delphes_card(self, path=None):
        if path == None:
            self._delphes_card = None
        else:
            self._delphes_card = path

    def set_parameters(self, set_param=None):
        if set_param == None:
            self._set_parameters = None
        else:
            out = ["set {}".format(i) for i in set_param]
            self._set_parameters = out

    def _default_input_file(self):
        self.import_model()
        self.define_multiparticles()
        self.process()
        self.add_process()
        self.output()
        self.launch()
        self.shower()
        self.detector()
        self.param_card()
        self.delphes_card()
        self.set_parameters()

    def show(self):
        """
        Print the current MG5InputFile
        """
        write = [
            self._import_model,
            self._define_multiparticles,
            self._process,
            self._add_process,
            self._output,
            self._launch,
            self._shower,
            self._detector,
            "0",
            self._param_card,
            self._delphes_card,
            self._set_parameters,
            "0",
        ]
        for w in write:
            if not (w == None):
                if isinstance(w, str):
                    print(w)
                elif isinstance(w, list):
                    [print(i) for i in w]

    def write(self):
        """
        Write a new madgraph script as MG5Script.txt used internally by madgraph.
        """
        write = [
            self._import_model,
            self._define_multiparticles,
            self._process,
            self._add_process,
            self._output,
            self._launch,
            self._shower,
            self._detector,
            "0",
            self._param_card,
            self._delphes_card,
            self._set_parameters,
            "0",
        ]
        f = open(os.path.join(self.work_dir, "MG5Script.txt"), "w+")
        for w in write:
            if not (w == None):
                if isinstance(w, str):
                    f.write(w + "\n")
                elif isinstance(w, list):
                    [f.write(i + "\n") for i in w]
        f.close()
        return


def read_mg_generation_info(file_path: str) -> dict:
    """
    Reads the MGGenerationInfo block in run tag banner (Saved by default with
    the name run_01_tag_1_banner.txt).

    Parameters:
        file_path: str
    Returns:
        mg5_gen_info: dict['number_of_events': int, 'cross_section_pb': float]
    """
    start_tag = "<MGGenerationInfo>"
    end_tag = "</MGGenerationInfo>"
    events_pattern = r"#\s*Number of Events\s*:\s*(\d+)"
    integrated_pattern = r"#\s*Integrated weight \(pb\)\s*:\s*(\d+\.\d+)"

    mg5_gen_info = {}

    with open(file_path, "r") as file:
        reading = False
        for line in file:
            line = line.strip()

            if line == start_tag:
                reading = True
            elif line == end_tag:
                break

            if reading:
                match = re.search(events_pattern, line)
                if match:
                    mg5_gen_info["number_of_events"] = int(match.group(1))
                match = re.search(integrated_pattern, line)
                if match:
                    mg5_gen_info["cross_section_pb"] = float(match.group(1))
            else:
                mg5_gen_info["number_of_events"] = None
                mg5_gen_info["cross_section_pb"] = None
    return mg5_gen_info


##########################################
# Class for reading a HiggsBounds output #
##########################################


class HiggsBoundsResults:
    def __init__(self, work_dir, model=None):
        self.work_dir = work_dir
        self.model = model

    def read(self, direct_path=None):
        """
        Read HiggsBounds_results.dat and outputs a dict with all the final results. For example for the BLSSM: \n
        [n, Mh(1), Mh(2), Mh(3), Mh(4), Mh(5), Mh(6), Mhplus(1), HBresult, chan, obsratio, ncomb]
        """
        if direct_path:
            file_path = direct_path
        else:
            file_path = os.path.join(self.work_dir, "HiggsBounds_results.dat")
        names = []
        values = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if ("HBresult" in line) & ("chan" in line):
                    names = line.split()[1:]

                    for subline in f:
                        values = subline.split()
        results = {}
        for n, v in zip(names, values):
            results[n] = float(v)

        return results

    def save(self, output_name, in_spheno_output=True):
        """Save the HiggsBounds results from the working directory to the Spheno output directory \n.
        To save all the higgs bounds results for scans for example."""


###########################################


class HiggsSignalsResults:
    def __init__(self, work_dir, model=None):
        self.work_dir = work_dir
        self.model = model

    def read(self, direct_path=None):
        """
        Read HiggsSignals_results.dat and outputs a dict all the results. For example for the BLSSM: \n
        [n, Mh(1), Mh(2), Mh(3), Mh(4), Mh(5), Mh(6), Mhplus(1), csq(mu), csq(mh), csq(tot), nobs(mu), nobs(mh), nobs(tot), Pvalue]
        """
        if direct_path:
            file_path = direct_path
        else:
            file_path = os.path.join(self.work_dir, "HiggsSignals_results.dat")
        names = []
        values = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if ("Mh(1)" in line) & ("Pvalue" in line):
                    names = line.split()[1:]

                    for subline in f:
                        values = subline.split()
        results = {}
        for n, v in zip(names, values):
            results[n] = float(v)

        return results

    def save(self, output_name, in_spheno_output=True):
        """
        Save the HiggsSignals results from the working directory to the
        Spheno output directory. To save all the higgs Signals results for
        scans for example.
        """
        if in_spheno_output:
            copy(
                os.path.join(self.work_dir, "HiggsSignals_results.dat"),
                os.path.join(
                    self.work_dir,
                    "SPheno" + self.model + "_output",
                    "HiggsSignals_results_" + str(output_name) + ".dat",
                ),
            )
        pass
