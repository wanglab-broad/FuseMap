import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle
import torch



def reset_parameters(para):
    torch.nn.init.xavier_uniform_(para)


class Discriminator(nn.Module):
    """
    Discriminator network for the FuseMap model.

    Parameters
    ----------
    latent_dim : int
        The dimension of the latent space.
    n_atlas : int
        The number of atlases.
    dropout_rate : float
        The dropout rate.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> disc = Discriminator(100, 10, 0.1)
    """

    def __init__(self, latent_dim, n_atlas, dropout_rate):
        super(Discriminator, self).__init__()

        self.linear_0 = nn.Linear(in_features=latent_dim, out_features=256, bias=True)
        self.act_0 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_0 = nn.Dropout(p=dropout_rate, inplace=False)

        self.linear_1 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.act_1 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_1 = nn.Dropout(p=dropout_rate, inplace=False)

        self.pred = nn.Linear(in_features=256, out_features=n_atlas, bias=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Discriminator class.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        
        Examples
        --------
        >>> x = torch.randn(10, 100)
        >>> disc = Discriminator(100, 10, 0.1)
        >>> y = disc(x)
        
        """

        x = self.linear_0(x)
        x = self.act_0(x)
        x = self.dropout_0(x)

        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x)

        x = self.pred(x)
        return x


class Adj_model(nn.Module):
    """
    Adjacency model for the FuseMap model.


    Parameters
    ----------
    N : int
        The number of nodes in the graph.

    Returns
    -------
    None

    Examples
    --------
    >>> adj = Adj_model(10)
    """

    def __init__(self, N):
        super(Adj_model, self).__init__()
        self.N = N
        # initialize your weight
        # self.weight = nn.Parameter(torch.full((N,N), 1.0/N))
        self.weight = nn.Parameter(torch.empty((N, N)))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self):
        """
        Forward pass for the Adj_model class.

        Returns
        -------
        weight_normalized : torch.Tensor
            The normalized weight matrix.

        Examples
        --------
        >>> adj = Adj_model(10)
        >>> adj()
        
        """

        weight_relu = torch.relu(self.weight)
        weight_relu = weight_relu + torch.eye(self.N).to(weight_relu.device)

        ### make weight matrix symmetric
        weight_upper = torch.triu(weight_relu)
        weight_lower = torch.tril(weight_relu).T
        weight_symmetric = torch.max(weight_upper, weight_lower)
        weight_symmetric = (
            weight_symmetric
            + weight_symmetric.T
            - torch.diag(weight_symmetric.diagonal())
        )

        k = 10  # torch.randint(3, 50, (1,)).item()
        topk, _ = torch.topk(weight_symmetric, k, dim=1)

        # Create a mask with 1s for the top k values and 0s for the rest
        mask = weight_symmetric >= topk[:, -1:]

        # Apply the mask to the weights
        weight_topk = weight_symmetric * mask.float()

        # Apply normalization along the row
        weight_sum = (
            torch.sum(weight_topk, dim=1, keepdim=True) + 1e-8
        )  # to prevent division by zero
        weight_normalized = weight_topk / weight_sum

        return weight_normalized


class FuseMapEncoder(nn.Module):
    """
    Encoder network for the FuseMap model.

    Parameters
    ----------
    input_dim : int
        The dimension of the input.
    hidden_dim : int
        The dimension of the hidden layer.
    latent_dim : int
        The dimension of the latent space.
    dropout_rate : float
        The dropout rate.
    normalization : str
        The normalization type.

    Returns
    -------
    None

    Examples
    --------
    >>> enc = FuseMapEncoder(100, 50, 10, 0.1)
    """

    def __init__(
        self, input_dim, hidden_dim, latent_dim, dropout_rate, normalization="batchnorm"
    ):
        super(FuseMapEncoder, self).__init__()
        self.dropout_0 = nn.Dropout(p=dropout_rate, inplace=False)
        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.activation_0 = nn.LeakyReLU(negative_slope=0.2)
        if normalization == "layernorm":
            self.bn_0 = nn.LayerNorm(hidden_dim, eps=1e-05)
        elif normalization == "batchnorm":
            self.bn_0 = nn.BatchNorm1d(
                hidden_dim,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

        self.dropout_1 = nn.Dropout(p=dropout_rate, inplace=False)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        if normalization == "layernorm":
            self.bn_1 = nn.LayerNorm(hidden_dim, eps=1e-05)
        if normalization == "batchnorm":
            self.bn_1 = nn.BatchNorm1d(
                hidden_dim,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, adj):
        """
        Forward pass for the FuseMapEncoder class.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        adj : torch.Tensor
            The adjacency matrix.
            
        Returns
        -------
        z_sample : torch.Tensor
        The sampled latent space tensor.
        None
        None
        z_mean : torch.Tensor
        The mean of the latent space tensor.
            
        Examples
        --------
        >>> x = torch.randn(10, 100)
        >>> adj = torch.randn(100, 100)
        >>> enc = FuseMapEncoder(100, 50, 10, 0.1)
        >>> z_sample, _, _, z_mean = enc(x, adj)
        """

        h_1 = self.linear_0(x)
        h_1 = self.bn_0(h_1)
        h_1 = self.activation_0(h_1)
        h_1 = self.dropout_0(h_1)

        h_2 = self.linear_1(h_1)
        h_2 = self.bn_1(h_2)
        h_2 = self.activation_1(h_2)
        h_2 = self.dropout_1(h_2)


        z_mean = self.mean(h_2)
        z_log_var = F.softplus(self.log_var(h_2))

        z_sample = D.Normal(z_mean, z_log_var)
        # z_sample_r = z_sample.rsample()

        z_spatial = torch.mm(adj.T, z_mean)

        return z_sample, None, z_spatial, z_mean


class FuseMapDecoder(nn.Module):
    def __init__(self, gene_embedding, var_index):
        super(FuseMapDecoder, self).__init__()
        self.gene_embedding = gene_embedding
        self.var_index = var_index
        # self.activation_3 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_3 = nn.ReLU()

    def forward(self, z_spatial, adj):
        h_4 = torch.mm(adj, z_spatial)
        x_recon_spatial = torch.mm(h_4, self.gene_embedding[:, self.var_index])
        x_recon_spatial = self.activation_3(x_recon_spatial)

        return x_recon_spatial


class FuseMapAdaptDecoder(nn.Module):
    def __init__(self, var_index, gene_embedding_pretrain, gene_embedding_new):
        super(FuseMapAdaptDecoder, self).__init__()
        self.gene_embedding_pretrain = gene_embedding_pretrain
        self.gene_embedding_new = gene_embedding_new
        self.var_index = var_index
        # self.activation_3 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_3 = nn.ReLU()
        
    def forward(self, z_spatial, adj, gene_embedding_pretrain, gene_embedding_new):
        h_4 = torch.mm(adj, z_spatial)
        # p=0
        # gene_embed_all = torch.hstack([gene_embedding_pretrain, self.gene_embedding_new ])

        gene_embed_all = torch.hstack(
            [
                self.gene_embedding_new,
                gene_embedding_pretrain,
            ]
        )

        x_recon_spatial = torch.mm(h_4, gene_embed_all[:, self.var_index])
        x_recon_spatial = self.activation_3(x_recon_spatial)

        return x_recon_spatial


class Fuse_network(nn.Module):
    """
    FuseMap model.
    
    Parameters
    ----------
    pca_dim : int
        The dimension of the PCA.
    input_dim : list
        The list of input dimensions.
    hidden_dim : int
        The dimension of the hidden layer.
    latent_dim : int
        The dimension of the latent space.
    dropout_rate : float
        The dropout rate.
    var_name : list
        The list of variable names.
    all_unique_genes : list
        The list of all unique genes.
    use_input : str
        The input type.
    n_atlas : int
        The number of atlases.
    input_identity : list
        The list of input identities.
    n_obs : list
        The list of number of observations.
    num_epoch : int
        The number of epochs.
    pretrain_model : bool
        Whether the model is pretrained.
    pretrain_n_atlas : int
        The number of pretrained atlases.
    PRETRAINED_GENE : list
        The list of pretrained genes.
    new_train_gene : list
        The list of new training genes.
    use_llm_gene_embedding : bool
        Whether to use the LLM gene embedding.

    Returns
    -------
    None

    Examples
    --------
    >>> model = Fuse_network(100, [10, 20], 50, 10, 0.1, ['gene1', 'gene2'], ['gene1', 'gene2'], 'norm', 2, ['scrna', 'scrna'], [100, 200], 100)
    
    """
    def __init__(
        self,
        pca_dim,
        input_dim,
        hidden_dim,
        latent_dim,
        dropout_rate,
        var_name,
        all_unique_genes,
        use_input,
        n_atlas,
        input_identity,
        n_obs,
        num_epoch,
        pretrain_model=False,
        pretrain_n_atlas=0,
        PRETRAINED_GENE=None,
        new_train_gene=None,
        use_llm_gene_embedding='false',
    ):
        super(Fuse_network, self).__init__()
        self.encoder = {}
        self.decoder = {}
        self.scrna_seq_adj = {}

        if use_input == "norm" or use_input == "raw":
            for i in range(n_atlas):
                self.add_encoder_module(
                    "atlas" + str(i), input_dim[i], hidden_dim, latent_dim, dropout_rate
                )
        self.encoder = nn.ModuleDict(self.encoder)

        ##### build gene embedding
        self.var_index = []
        if use_llm_gene_embedding=='false':
            if pretrain_model:
                self.gene_embedding_pretrained = nn.Parameter(
                    torch.zeros(latent_dim, len(PRETRAINED_GENE))
                )
                self.gene_embedding_new = nn.Parameter(
                    torch.zeros(latent_dim, len(new_train_gene))
                )
                all_genes = new_train_gene + PRETRAINED_GENE
                for ij in range(n_atlas):
                    self.var_index.append([all_genes.index(i) for i in var_name[ij]])
                reset_parameters(self.gene_embedding_new)
            else:
                self.gene_embedding = nn.Parameter(
                    torch.zeros(latent_dim, len(all_unique_genes))
                )
                for ij in range(n_atlas):
                    self.var_index.append(
                        [all_unique_genes.index(i) for i in var_name[ij]]
                    )
                reset_parameters(self.gene_embedding)

        elif use_llm_gene_embedding=='combine':
            if pretrain_model:
                self.gene_embedding_pretrained = nn.Parameter(
                    torch.zeros(latent_dim, len(PRETRAINED_GENE))
                )
                self.gene_embedding_new = nn.Parameter(
                    torch.zeros(latent_dim, len(new_train_gene))
                )
                all_genes = new_train_gene + PRETRAINED_GENE
                for ij in range(n_atlas):
                    self.var_index.append([all_genes.index(i) for i in var_name[ij]])
                reset_parameters(self.gene_embedding_new)
            
                path_genept="./data/GenePT_gene_protein_embedding_model_3_text_pca.pickle"
                with open(path_genept, "rb") as fp:
                    GPT_3_5_gene_embeddings = pickle.load(fp)  

                self.llm_gene_embedding = torch.zeros(latent_dim, len(all_genes))    
                for i,gene in enumerate(all_genes):
                    if gene in GPT_3_5_gene_embeddings.keys():
                        self.llm_gene_embedding[:,i] = torch.tensor(GPT_3_5_gene_embeddings[gene])

                # Calculate gene embedding loss
                ground_truth_matrix = self.llm_gene_embedding.T
                ind = torch.sum(ground_truth_matrix,axis=1)!=0
                ground_truth_matrix=ground_truth_matrix[ind,:]
                
                self.llm_ind=ind
                ground_truth_matrix_normalized = ground_truth_matrix / ground_truth_matrix.norm(dim=1, keepdim=True)
                self.ground_truth_rel_matrix = torch.matmul(ground_truth_matrix_normalized, ground_truth_matrix_normalized.T)

            else:
                self.gene_embedding = nn.Parameter(
                    torch.zeros(latent_dim, len(all_unique_genes))
                )
                for ij in range(n_atlas):
                    self.var_index.append(
                        [all_unique_genes.index(i) for i in var_name[ij]]
                    )
                reset_parameters(self.gene_embedding)

                path_genept="./data/GenePT_gene_protein_embedding_model_3_text_pca.pickle"
                with open(path_genept, "rb") as fp:
                    GPT_3_5_gene_embeddings = pickle.load(fp)  

                self.llm_gene_embedding = torch.zeros(latent_dim, len(all_unique_genes))    
                for i,gene in enumerate(all_unique_genes):
                    if gene in GPT_3_5_gene_embeddings.keys():
                        self.llm_gene_embedding[:,i] = torch.tensor(GPT_3_5_gene_embeddings[gene])

                # Calculate gene embedding loss
                ground_truth_matrix = self.llm_gene_embedding.T
                ind = torch.sum(ground_truth_matrix,axis=1)!=0
                ground_truth_matrix=ground_truth_matrix[ind,:]
                
                self.llm_ind=ind
                ground_truth_matrix_normalized = ground_truth_matrix / ground_truth_matrix.norm(dim=1, keepdim=True)
                self.ground_truth_rel_matrix = torch.matmul(ground_truth_matrix_normalized, ground_truth_matrix_normalized.T)

        elif use_llm_gene_embedding=='true':
            if pretrain_model:
                raise ValueError("pretrain_model is not supported for use_llm_gene_embedding")
            else:
                self.gene_embedding =  torch.zeros(latent_dim, len(all_unique_genes))
                for ij in range(n_atlas):
                    self.var_index.append(
                        [all_unique_genes.index(i) for i in var_name[ij]]
                    )

                path_genept="./jupyter_notebook/data/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text_pca.pickle"
                with open(path_genept, "rb") as fp:
                    GPT_3_5_gene_embeddings = pickle.load(fp)    
                # reset_parameters(self.gene_embedding)
                # ind=0
                for i,gene in enumerate(all_unique_genes):
                    if gene in GPT_3_5_gene_embeddings.keys():
                        # print(gene)
                        # ind+=1
                        self.gene_embedding[:,i] = torch.tensor(GPT_3_5_gene_embeddings[gene])
                self.gene_embedding=nn.Parameter(self.gene_embedding)
                self.gene_embedding.requires_grad = False
        else:
            raise ValueError("use_llm_gene_embedding should be either 'true' or 'false' or 'combine'")

        ##### build decoders
        if pretrain_model:
            for ij in range(n_atlas):
                self.add_adaptdecoder_module(
                    "atlas" + str(ij),
                    self.var_index[ij],
                    self.gene_embedding_pretrained,
                    self.gene_embedding_new,
                )
        else:
            for ij in range(n_atlas):
                self.add_decoder_module(
                    "atlas" + str(ij),
                    self.gene_embedding,
                    self.var_index[ij],
                )
        self.decoder = nn.ModuleDict(self.decoder)

        ##### build discriminators
        self.discriminator_single = Discriminator(latent_dim, n_atlas, dropout_rate)
        self.discriminator_spatial = Discriminator(latent_dim, n_atlas, dropout_rate)

        if pretrain_model:
            self.discriminator_single_pretrain = Discriminator(
                latent_dim, pretrain_n_atlas, dropout_rate
            )
            self.discriminator_spatial_pretrain = Discriminator(
                latent_dim, pretrain_n_atlas, dropout_rate
            )

        ##### build scrnaseq adjacency matrix
        for i in range(n_atlas):
            if input_identity[i] == "scrna":
                self.scrna_seq_adj["atlas" + str(i)] = Adj_model(n_obs[i])
        self.scrna_seq_adj = nn.ModuleDict(self.scrna_seq_adj)

    def add_encoder_module(
        self, key, input_dim, hidden_dim, latent_dim, dropout_rate=0.1
    ):
        """
        Add an encoder module to the model.

        Parameters
        ----------
        key : str
            The key for the encoder module.
        input_dim : int
            The dimension of the input.
        hidden_dim : int
            The dimension of the hidden layer.
        latent_dim : int
            The dimension of the latent space.
        dropout_rate : float
            The dropout rate.

        Returns
        -------
        None

        Examples
        --------
        >>> model = Fuse_network(100, [10, 20], 50, 10, 0.1, ['gene1', 'gene2'], ['gene1', 'gene2'], 'norm', 2, ['scrna', 'scrna'], [100, 200], 100)
        >>> model.add_encoder_module('atlas1', 10, 50, 10, 0.1)

        """
        self.encoder[key] = FuseMapEncoder(
            input_dim, hidden_dim, latent_dim, dropout_rate
        )

    def add_decoder_module(self, key, gene_embedding, var_index):
        """
        Add a decoder module to the model.

        Parameters
        ----------
        key : str
            The key for the decoder module.
        gene_embedding : torch.Tensor
            The gene embedding tensor.
        var_index : list
            The list of variable indices.

        Returns
        -------
        None

        Examples
        --------
        >>> model = Fuse_network(100, [10, 20], 50, 10, 0.1, ['gene1', 'gene2'], ['gene1', 'gene2'], 'norm', 2, ['scrna', 'scrna'], [100, 200], 100)
        >>> model.add_decoder_module('atlas1', torch.randn(10, 100), [1, 2, 3])

        """
        self.decoder[key] = FuseMapDecoder(gene_embedding, var_index)

    def add_adaptdecoder_module(self, key, var_index, gene_pretrain, gene_new):
        """
        Add an adapted decoder module to the model.

        Parameters
        ----------
        key : str
            The key for the adapted decoder module.
        var_index : list
            The list of variable indices.
        gene_pretrain : torch.Tensor
            The pretrained gene embedding tensor.
        gene_new : torch.Tensor
            The new gene embedding tensor.

        Returns
        -------
        None

        Examples
        --------
        >>> model = Fuse_network(100, [10, 20], 50, 10, 0.1, ['gene1', 'gene2'], ['gene1', 'gene2'], 'norm', 2, ['scrna', 'scrna'], [100, 200], 100)
        >>> model.add_adaptdecoder_module('atlas1', [1, 2, 3], torch.randn(10, 100), torch.randn(10, 100))

        """
        self.decoder[key] = FuseMapAdaptDecoder(var_index, gene_pretrain, gene_new)




class NNTransfer(nn.Module):
    def __init__(self, input_dim=128, output_dim=10):
        super(NNTransfer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x= self.activate(x)
        return x