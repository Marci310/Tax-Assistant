import pytest
from langchain_core.documents import Document
from fastapi.testclient import TestClient

from api.dependencies import get_qa_service
from api.services.qa_services import QAService
from core.config import Settings
from main import app


class StubVectorStore:
    def __init__(self, hits: list[tuple[Document, float]]):
        self.hits = hits
        self.queries: list[tuple[str, int]] = []

    async def asimilarity_search_with_score(self, query: str, k: int = 4):
        self.queries.append((query, k))
        return self.hits[:k]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="ollama",
        min_similarity=0.45,
        top_k=4,
        llm_timeout_seconds=5,
        max_concurrent_llm_calls=2,
        ingest_on_startup=False,
    )


@pytest.fixture
def documents() -> list[Document]:
    return list(DOCUMENTS)


@pytest.fixture
def make_vector_store():
    def _make(hits: list[tuple[Document, float]]) -> StubVectorStore:
        return StubVectorStore(hits)
    return _make


DOCUMENTS: list[Document] = [
    Document(page_content="Nem kell ilyen igazolással rendelkeznie azon adózónak, aki a Nemzeti Összefogás Számlaszáma javára juttat adományt.\n\nA felsőoktatási támogatási megállapodás alapján a felsőoktatási intézménynek visszafizetési kötelezettség nélkül adott támogatások, juttatások, térítés nélküli eszközátadások, térítés nélküli szolgáltatásnyújtások összege mellett az intézmény fenntartójának nyújtott – előzőek szerint meghatározott – ingyenes juttatások összege is elismert költség.",metadata={"source":"41 A társasági adó legfontosabb szabályai 20260202.pdf","booklet":"41","title":"A társasági adó legfontosabb szabályai","published":"2026-02-02","start_index":32713,"page":18,"page_end":18}),
    Document(page_content="Az elszámolható költség\n− az energiahatékonysági beruházás, felújítás célját szolgáló, a kizárólag a magasabb energiahatékonysági szint elérését elősegítő tárgyi eszköz, immateriális jószág bekerülési értéke, értékének növekedése, ha ezen beruházás, felújítás egyértelműen beazonosítható, és nincs olyan kevésbé energiahatékony alternatív beruházás, felújítás, amely az adókedvezmény vagy más állami támogatás hiányában megvalósulna, vagy\n− a magasabb energiahatékonysági szint eléréséhez szükséges beruházás, felújítás elszámolható költsége és az alternatív beruházás, felújítás költsége közötti különbözet (többletköltség).\n\nAz adókedvezmény adózónként és beruházásonként, felújításonként nem haladhatja meg jelenértéken az elszámolható költségek\n− a) Budapesten 30 százalékának,\n− b) Észak-Magyarország, Észak-Alföld, Dél-Alföld, Dél-Dunántúl, Közép-Dunántúl, Nyugat-Dunántúl vagy Pest régióban 45 százalékának megfelelő mértéket, de legfeljebb a 30 millió eurónak megfelelő forintösszeget.",metadata={"source":"41 A társasági adó legfontosabb szabályai 20260202.pdf","booklet":"41","title":"A társasági adó legfontosabb szabályai","published":"2026-02-02","start_index":67501,"page":33,"page_end":33}),
    Document(page_content="Tehát „B” cég a vele kapcsolt vállalkozási viszonyban álló „A” cégnek – vagy akár több kapcsolt vállalkozásának – átadhatja az őt megillető kedvezményt vagy annak egy részét.\n\nÍgy „A” cégnél a csökkentő tétel akkor alkalmazható, ha rendelkezik az adóbevallás benyújtásáig a kapcsolt vállalkozása, vagyis a „B” cég írásos nyilatkozatával, amely tartalmazza a kapcsolt vállalkozás adóévi saját tevékenységi körében végzett K+F tevékenysége közvetlen költségének összegét, valamint az összeg után az adózó(k) által érvényesíthető összeget. A nyilatkozat(ok)ban meghatározott összeg(ek) nem haladhatják meg a nyilatkozat kiállítója által adóalap-csökkentő tételként érvényesíthető összeget. A nyilatkozatban foglaltak teljesüléséért az adózó és kapcsolt vállalkozása egyetemlegesen felel. A nyilatkozat kiállítója, valamint a kedvezményezett a nyilatkozatban szereplő adatokról a társaságiadó-bevallásban adatszolgáltatásra kötelezett.",metadata={"source":"41 A társasági adó legfontosabb szabályai 20260202.pdf","booklet":"41","title":"A társasági adó legfontosabb szabályai","published":"2026-02-02","start_index":23140,"page":14,"page_end":14})
]

@pytest.fixture
def make_client(settings):
    def _make(graph, test_settings: Settings | None = None) -> TestClient:
        service_settings = test_settings or settings
        app.dependency_overrides[get_qa_service] = lambda: QAService(service_settings, graph)
        return TestClient(app, raise_server_exceptions=False)

    yield _make
    app.dependency_overrides.clear()
