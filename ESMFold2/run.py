

# Built-in modules
import time
import string
import pathlib
import argparse
from typing import Callable
from functools import wraps

# Third-party modules
import torch
import esm.sdk
import numpy as np
from tqdm import tqdm
from esm.models.esmfold2 import ESMFold2InputBuilder
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# Custom libraries
from fasta import Fasta

# Argparse
parser = argparse.ArgumentParser(description='ESMFold')
parser.add_argument('--fasta', type=str, required=True, 
                    help='Can be either a FASTA file or a directory with FASTA files.' \
                        'If a file, the sequences in the file will be folded as chains of a single structure.' \
                        'If a directory, the sequences in each file will be folded as separate structures.')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Output directory')
parser.add_argument('--name', type=str, required=False,
                    help='Name of the output file(s). Does not need to be provided if --fasta is a directory,' \
                    'in which case the name of the FASTA file will be used' )
parser.add_argument('--biohub', action='store_true',
                    help='Whether to use the Biohub API for folding. If not set, the folding will be done locally using the ESMFold2Model from huggingface. Note that using the Biohub API requires an API key and may incur costs depending on usage.')
parser.add_argument('--fast', action='store_true',
                    help='Whether to use the fast version of ESMFold2. The fast version is a smaller model that runs faster but may be less accurate than the standard version. Note that the fast version is only available through the Biohub API and not as a local model on huggingface.')
args = parser.parse_args()

class ESMFold2:

    # Biohub API key
    API_KEY = '7aTVRVsat7XAHSdH6a191f'

###############################################################################
#######                         INSTANTIATION                           #######
###############################################################################

    def __init__(
        self, 
        biohub: bool = False,
        fast: bool = False
    ):
        if biohub:
            self.model = self._load_from_biohub(fast)
        else:
            self.model = self._load_from_huggingface(fast)
        self.config = None
        self.result = None

    def _load_from_biohub(
        self, 
        fast: bool
    ) -> 'SequenceStructureForgeInferenceClient':
        '''
        Load ESMFold2 model from Biohub API.

        Parameters
        ----------
        fast : bool
            Whether to use the fast version of ESMFold2.

        Returns
        -------
        esm.sdk.esmfold2_client.SequenceStructureForgeInferenceClient
            The loaded ESMFold2 client.
        '''
        model = 'esmfold2-fast-2026-05' if fast else 'esmfold2-2026-05'
        client = esm.sdk.esmfold2_client(
            model=model,
            url="https://biohub.ai",
            token=self.API_KEY
        )
        return client

    def _load_from_huggingface(
        self, 
        fast: bool
    ) -> ESMFold2Model:
        '''
        Load ESMFold2 model from huggingface.

        Parameters
        ----------
        fast : bool
            Whether to use the fast version of ESMFold2.

        Returns
        -------
        ESMFold2Model
            The loaded ESMFold2 model.
        '''
        name = 'biohub/ESMFold2-Fast' if fast else 'biohub/ESMFold2'
        model = ESMFold2Model.from_pretrained(name).cuda().eval()
        return model

###############################################################################
#######                        CONFIGURATION                            #######
###############################################################################

    def set_config(
        self,
        num_loops: int = 3,
        num_sampling_steps: int = 32,
        include_pae: bool = True,
        include_distogram: bool = True,
        include_pair_chains_iptm: bool = True,
        include_embeddings: bool = False
    ):
        '''
        Set configuration.

        Parameters
        ----------
        num_loops : int, optional
            Number of loops during iterative refinement, by default 3. 
            Increasing number of loops may improve accuracy but definitely
            increases runtime.
        num_sampling_steps : int, optional
            Number of sampling steps during diffusion sampling, by default 32.and
            Increasing number of sampling steps may give more refined
            structures but also increases runtime.
        include_pae : bool, optional
            Whether to report PAE, by default True
        include_distogram : bool, optional
            Whether to report distogram, by default True
        include_pair_chains_iptm : bool, optional
            Whether to report pair chains iptm, by default True
        include_embeddings : bool, optional
            Whether to report embeddings, by default False
        '''
        self.config = esm.sdk.api.FoldingConfig(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            include_pae=include_pae,
            include_distogram=include_distogram,
            include_pair_chains_iptm=include_pair_chains_iptm,
            include_embeddings=include_embeddings
        )

###############################################################################
#######                             FOLDING                             #######
###############################################################################
 
    def timer(function: Callable) -> Callable: 
        '''
        Measures the time taken to fold a sequence.

        Parameters
        ----------
        function : Callable
            fold function

        Returns
        -------
        Callable
            Wrapper function
        '''
        @wraps(function)
        def wrapper(*args, **kwargs): 
            t1 = time.time() 
            result = function(*args, **kwargs) 
            t2 = time.time() 
            print(f'Sequence folded in {(t2-t1):.4f}s') 
            return result 
        return wrapper 

    @timer
    def fold(
        self, 
        seq2molecule: list[tuple[str, str]]
    ) -> None:
        '''
        Fold sequences into structures.

        Parameters
        ----------
        seq2molecule : list[tuple[str, str]]
            A list of tuples mapping sequences to molecule types. Supported 
            molecule types are 'protein', 'rna', 'dna', and 'ligand'.
            For ligands, the sequence should be a SMILES string. 
            Note that protein modifications are also possible with ESMFold2 but
            currently not supported in this wrapper.

        Raises
        ------
        ValueError
            If an unsupported molecule type is provided.
        '''
        # Prepare folding inputs
        folding_input = []
        for idx, (seq, molecule) in enumerate(seq2molecule):
            id = string.ascii_uppercase[idx]
            if molecule == 'protein':
                protein = esm.utils.structure.input_builder.ProteinInput(
                    id=id, 
                    sequence=seq
                )
                folding_input.append(protein)
            elif molecule == 'rna':
                rna = esm.utils.structure.input_builder.RNAInput(
                    id=id, 
                    sequence=seq
                )
                folding_input.append(rna)
            elif molecule == 'dna':
                dna = esm.utils.structure.input_builder.DNAInput(
                    id=id, 
                    sequence=seq
                )
                folding_input.append(dna)
            elif molecule == 'ligand':
                ligand = esm.utils.structure.input_builder.LigandInput(
                    id=id,
                    smiles=seq
                )
                folding_input.append(ligand)
            else:
                raise ValueError(f"Unsupported molecule type: {molecule}")

        # Create prediction input
        prediction_input = esm.utils.structure.input_builder.StructurePredictionInput(
            sequences=folding_input
        )
        
        # Fold the structure
        if  self.model.__class__.__name__ == 'SequenceStructureForgeInferenceClient':
            result = self.model.fold_all_atom(
                prediction_input, 
                config=self.config
            )
        elif self.model.__class__.__name__ == 'ESMFold2Model':
            result = ESMFold2InputBuilder().fold(
                self.model, 
                prediction_input,
                num_loops=self.config.num_loops,
                num_sampling_steps=self.config.num_sampling_steps,
                #include_pae=self.config.include_pae,
                #include_distogram=self.config.include_distogram,
                #include_pair_chains_iptm=self.config.include_pair_chains_iptm,
                #include_embeddings=self.config.include_embeddings,
                seed=0
            )

        self.result = result
        
###############################################################################
#######                         PROPERTIES                              #######
###############################################################################
 
    @property
    def structure(self) -> str:
        if self.result is None:
            raise ValueError("No folding result available. Please run fold() first.")
        return self.result.complex.to_mmcif()
    
    @property
    def plddt(self) -> np.ndarray:
        if self.result is None:
            raise ValueError("No folding result available. Please run fold() first.")
        return self.result.plddt
    
    @property
    def pae(self) -> np.ndarray:
        if self.result is None:
            raise ValueError("No folding result available. Please run fold() first.")
        return self.result.pae
    
    @property
    def distance_map(self) -> np.ndarray:
        '''
        A distogram is a matrix of shape (N, N, B), where N is the number of 
        protein residues and B is the number of distance bins. Each entry
        (i, j, k) in the distogram represents the probability that the distance
        between residues i and j falls into the k-th distance bin. The distance
        bins are defined by a set of thresholds, which are typically determined
        based on the distribution of distances observed in known protein 
        structures (e.g. 0-4Å, 4-8Å, 8-12Å, etc.). 
        Predicted distance (or contact) maps can be derived from the distogram
        by applying softmax over the distance bins and calculating the expected
        distance for each residue pair.

        Returns
        -------
        np.ndarray
            The predicted distogram.
        '''
        # Validation
        if self.result is None:
            raise ValueError("No folding result available. Please run fold() first.")
        # Compute dmap from distogram
        bins = np.append(0,np.linspace(2.3125,21.6875,63))
        distogram = torch.softmax(self.result.distogram, dim=-1)
        distance_map = (distogram * bins).sum(dim=-1).cpu().numpy()
        return distance_map
    
    @property
    def ptm(self) -> float:
        if self.result is None:
            raise ValueError("No folding result available. Please run fold() first.")
        return self.result.ptm
    
    @property
    def iptm(self) -> float:
        if self.result is None:
            raise ValueError("No folding result available. Please run fold() first.")
        return self.result.pair_chains_iptm

# Single or multiple FASTA files
if args.fasta.endswith('.fasta') or args.fasta.endswith('.fa'):
    fastas = [args.fasta]
    file_names = [args.name]
elif pathlib.Path(args.fasta).is_dir():
    sort_key = lambda x: int(x.stem)
    try: 
        fastas = sorted([fasta_file for fasta_file in pathlib.Path(args.fasta).glob('*.fasta')], key=sort_key)
    except ValueError:
        fastas = sorted([fasta_file for fasta_file in pathlib.Path(args.fasta).glob('*.fasta')])
    file_names = [fasta_file.stem for fasta_file in fastas]
    print(f'Folding {len(fastas)} sequences from {args.fasta} directory')
else:
    raise ValueError(f'Invalid input: {args.fasta} is neither a file nor a directory.')

# Fold sequences
esmfold2 = ESMFold2(
    biohub=args.biohub,
    fast=args.fast
)
esmfold2.set_config(
    num_loops=3,
    num_sampling_steps=32
)
for fasta, file_name in tqdm(zip(fastas, file_names), total=len(fastas)):
    fasta = Fasta.from_file(fasta)
    seq2molecule = [(sequence, 'protein') for sequence in fasta.sequences]
    esmfold2.fold(seq2molecule)

    # Save structure
    with open(f'{args.out_dir}/{file_name}.esmfold2.cif', 'w') as f:
            f.write(esmfold2.structure)
    # Save pLDDT
    np.save(f'{args.out_dir}/{file_name}.esmfold2.plddt', esmfold2.plddt)
    # Save PAE
    np.save(f'{args.out_dir}/{file_name}.esmfold2.pae', esmfold2.pae)
    # Save contact map
    np.save(f'{args.out_dir}/{file_name}.esmfold2.dmap', esmfold2.distance_map)
    # Save pTM and ipTM
    np.save(f'{args.out_dir}/{file_name}.esmfold2.ptm', esmfold2.ptm)
    np.save(f'{args.out_dir}/{file_name}.esmfold2.iptm', esmfold2.iptm)
