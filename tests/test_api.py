import io


def test_health(client):
    assert client.get("/api/health").json()["models_loaded"] is True


def test_chat_anonymous(client):
    r = client.post("/api/chat", json={"question": "Как оформить академический отпуск?"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "ru" and body["sources"]


def test_validation(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "abc", "mode": "drop table"}).status_code == 422


def test_register_login_history_feedback(client):
    r = client.post("/api/auth/register", json={"email": "student1@su.kz", "password": "secret12"})
    assert r.status_code == 200
    h = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert client.post("/api/auth/register", json={"email": "student1@su.kz", "password": "secret12"}).status_code == 400
    assert client.post("/api/auth/login", json={"email": "student1@su.kz", "password": "wrong"}).status_code == 401
    m = client.post("/api/chat", json={"question": "Жатақханаға қандай құжаттар қажет?"}, headers=h).json()
    assert client.post(f"/api/messages/{m['message_id']}/feedback", json={"value": 1}).json()["ok"]
    conv = client.get("/api/conversations", headers=h).json()
    assert conv and conv[0]["messages"] == 1
    assert client.get("/api/admin/stats", headers=h).status_code == 403


def admin_headers(client):
    r = client.post("/api/auth/login", json={"email": "admin@satbayev.university", "password": "Admin12345!"})
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def test_admin_stats_and_upload(client):
    h = admin_headers(client)
    s = client.get("/api/admin/stats", headers=h).json()
    assert s["documents"]["total"] >= 14
    text = ("Инструкция по работе с электронной библиотекой.\n\nЭлектронная библиотека доступна студентам "
            "круглосуточно через личный кабинет. Для входа используется логин и пароль студента университета.")
    r = client.post("/api/documents", headers=h, data={"title": "Инструкция по работе с Электронной библиотекой"},
                    files={"file": ("lib.txt", io.BytesIO(text.encode()), "text/plain")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    a = client.post("/api/chat", json={"question": "Электронная библиотека доступна через личный кабинет?",
                                       "top_k": 10}).json()
    assert doc_id in [x["doc_id"] for x in a["retrieved"]]
    assert client.delete(f"/api/documents/{doc_id}", headers=h).json()["deleted_chunks"] == 1
    assert client.get("/api/admin/logs", headers=h).json()


def test_upload_forbidden_for_anonymous(client):
    r = client.post("/api/documents", data={"title": "x"}, files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 401


def test_frontend_served(client):
    assert "Satbayev" in client.get("/").text
