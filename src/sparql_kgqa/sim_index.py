import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score


class SimilarityIndex:
    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1",
            device=device
        )
        self.dim = self.model.get_sentence_embedding_dimension() or 1024
        self.embeddings = torch.empty(0, self.dim, device=self.device)
        self.samples = []

    def save(
        self,
        path: str
    ) -> None:
        torch.save(
            {
                "embeddings": self.embeddings,
                "samples": self.samples
            },
            path
        )

    def load(
        self,
        path: str | list[str]
    ) -> None:
        if isinstance(path, str):
            path = [path]

        self.samples = []
        self.embeddings = torch.empty(0, self.dim, device=self.device)

        for p in path:
            checkpoint = torch.load(p, map_location=self.device)
            self.embeddings = torch.cat([
                self.embeddings,
                checkpoint["embeddings"]
            ])
            self.samples.extend(checkpoint["samples"])

    def add(
        self,
        samples: list[tuple[str, str]],
        batch_size: int = 32,
        progress: bool = False
    ) -> None:
        embeddings = self.model.encode(
            [sample[0] for sample in samples],
            show_progress_bar=progress,
            batch_size=batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        assert isinstance(embeddings, torch.Tensor)
        self.embeddings = torch.cat([
            self.embeddings,
            embeddings.to(self.device)
        ])
        self.samples.extend(samples)

    def find_matches(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> list[tuple[str, str]]:
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        similarities = dot_score(
            query_embedding.to(self.device),
            self.embeddings
        )
        top_k_samples = torch.topk(
            similarities[0],
            min(top_k, len(self.embeddings))
        )
        return [
            self.samples[idx]
            for idx, sim in zip(
                top_k_samples.indices,
                top_k_samples.values
            )
            if sim >= threshold
        ]
