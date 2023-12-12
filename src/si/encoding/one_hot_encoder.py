import numpy as np

class OneHotEncoder:
    """
    One-hot encoding is a representation technique where categorical data, such as words in a text sequence (or
    characters in a sequence), is converted into binary vectors with only one element set to 1 (indicating the
    presence of a specific category) and the rest set to 0.
    """

    def __init__(self, padder:str, max_length: int = None):
        """
        Parameters
        ----------
        padder:
            character to perform padding with
        max_length:
            maximum length of sequences
        
        Attributes
        -----------
        alphabet:
            the unique characters in the sequences
        char_to_index:dict
            dictionary mapping characters in the alphabet to unique integers
        index_to_char:dict
            reverse of char_to_index (dictionary mapping integers to characters)
        """
        # arguments
        self.padder = padder
        self.max_lenght = max_length

        #estimated parameters
        self.alphabet=set()
        self.char_to_index={}
        self.index_to_char={}
    
    def fit (self, data: list[str])->'OneHotEncoder':
        """
        Fits to the dataset.

        Parameters
        ---------
        data:list[str]
            list of sequences to learn from
        -------

        """
        
        if self.max_lenght is None:
            lengths = []
            for sequence in data:
                lengh = len(sequence)
                lengths.append(lengh)
            self.max_lenght = np.max(lengths) 
    
        all_seq= "".join(data) 
        self.alphabet = np.unique(list(all_seq)) 

        
        indexes = np.arange(1, len(self.alphabet) + 1) 
        
        self.char_to_index = dict(zip(self.alphabet, indexes)) 
        self.index_to_char = dict(zip(indexes,self.alphabet)) 


        #special padding character
        if self.padder not in self.alphabet:
            self.alphabet = np.append(self.alphabet, self.padder)  
            max_index = max(self.char_to_index.values())  
            new_index = max_index + 1 
            self.char_to_index[self.padder] = new_index 
            self.index_to_char[new_index] = self.padder 
        
                
        return self
    
    
    def transform(self, data:list[str]) ->np.ndarray:
        """
        Parameter
        ---------
        data:list[str]
            data to encode
        
        Returns
        --------
        np.ndarray:
            One-hot encoded matrices
        """
    
        sequence_trim_pad = []
        for sequence in data:
            trim_pad = sequence[:self.max_lenght].ljust(self.max_lenght, self.padder)
            sequence_trim_pad.append(trim_pad)
        
        
        
        one_hot_encode = []
        identity_matrix =np.eye(len(self.alphabet)) 
        print(identity_matrix)
        
        for adjusted_seq in sequence_trim_pad: 
            for letter in adjusted_seq: 
                value_in_dict = self.char_to_index.get(letter)
                one_hot_sequence = identity_matrix[value_in_dict - 1] 
                
                one_hot_encode.append(one_hot_sequence)
        return one_hot_encode

    def fit_transform(self, data: list[str]) -> np.ndarray:
        """
        Parameters
        ----------
        data: list[str]
            list of sequences to learn from
        Returns
        -------
        np.ndarray:
            One-hot encoded matrices
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> list[str]:
        """
        Parameters
        ----------
        data: np.ndarray-vem de cima
            one-hot encoded matrices to decode
        Returns
        -------
        list[str]:
            Decoded sequences
        """
        
        index = []
        for one_hot_matrix in data:
            indexes = np.argmax(one_hot_matrix)  
            index.append(indexes)  

        total_sequences = []
        for each_index in index:
            char = self.index_to_char.get(each_index + 1) 
            total_sequences.append(char)
            text ="".join(total_sequences) 
        
        trimmed_segments = []
        for i in range (0,len(text),self.max_lenght):
            string = text[i:i + self.max_lenght]
            trimmed_string = string.rstrip(self.padder) 
            trimmed_segments.append(trimmed_string)
        return trimmed_segments 


if __name__ == '__main__':
    sequences = ["oes", "daofm", "faofn"]
    padder = '*'
    
    encoder = OneHotEncoder(padder="?", max_length=10)

    encoed_data = encoder.fit_transform(sequences)

    print("Alphabet:", encoder.alphabet)
    print("Char to Index:", encoder.char_to_index)
    print("Index to Char:", encoder.index_to_char)
    print("Encoded Sequences:")
    print(encoed_data)

    decoded_sequences = encoder.inverse_transform(encoed_data)
    print("Decoded Sequences:")
    print(decoded_sequences)
