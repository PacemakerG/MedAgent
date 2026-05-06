def test_auth_me_default_anonymous(test_client):
    res = test_client.get("/api/v1/auth/me")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["logged_in"] is False
    assert data["user_id"] == "anonymous"


def test_auth_login_and_logout(test_client):
    login = test_client.post(
        "/api/v1/auth/login",
        json={
            "user_id": "doctor_zhang",
            "tenant_id": "hospital_a",
            "password": "strong-password-123",
        },
    )
    assert login.status_code == 200
    payload = login.json()
    assert payload["logged_in"] is True
    assert payload["user_id"] == "doctor_zhang"
    assert payload["tenant_id"] == "hospital_a"
    assert payload["access_token"]

    me = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    assert me.status_code == 200
    me_data = me.json()
    assert me_data["logged_in"] is True
    assert me_data["user_id"] == "doctor_zhang"
    assert me_data["tenant_id"] == "hospital_a"

    logout = test_client.post("/api/v1/auth/logout")
    assert logout.status_code == 200
    assert logout.json()["logged_in"] is False

    me_after = test_client.get("/api/v1/auth/me")
    assert me_after.status_code == 200
    assert me_after.json()["logged_in"] is False


def test_auth_login_rejects_wrong_password(test_client):
    user_id = "doctor_password_check"
    created = test_client.post(
        "/api/v1/auth/login",
        json={
            "user_id": user_id,
            "tenant_id": "hospital_a",
            "password": "first-password-123",
        },
    )
    assert created.status_code == 200

    rejected = test_client.post(
        "/api/v1/auth/login",
        json={
            "user_id": user_id,
            "tenant_id": "hospital_a",
            "password": "wrong-password-456",
        },
    )
    assert rejected.status_code == 401
