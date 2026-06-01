from qdrant_client import QdrantClient

# SETTINGS
COLLECTION = "tryface"   # change if needed
LIMIT = 1000             # how many points to read

client = QdrantClient(host="localhost", port=6333)


def main():
    print("\n==============================")
    print("QDRANT DATASET INSPECTION")
    print("==============================")

    # 1️ Get collection info
    try:
        info = client.get_collection(COLLECTION)
        print("\nCollection Info:")
        print(info)
    except Exception as e:
        print(" Failed to get collection info:", e)
        return

    # 2️ Scroll through stored points
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION,
            limit=LIMIT,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(" Failed to scroll collection:", e)
        return

    print("\nTotal points retrieved:", len(points))

    # 3️ Extract unique names
    unique_names = set()
    payload_keys = set()

    for p in points:
        payload = p.payload or {}

        
        for k in payload.keys():
            payload_keys.add(k)

        
        name = (
            payload.get("student_name")
            or payload.get("name")
            or payload.get("label")
            or "NO_NAME_FOUND"
        )

        unique_names.add(name)

    # 4️ Print results
    print("\n==============================")
    print("UNIQUE NAMES IN DATASET:")
    print("==============================")
    for n in sorted(unique_names):
        print("-", n)

    print("\nTotal unique names:", len(unique_names))

    print("\n==============================")
    print("PAYLOAD KEYS FOUND:")
    print("==============================")
    for k in sorted(payload_keys):
        print("-", k)

    # 5️ Show sample payloads
    print("\n==============================")
    print("FIRST 10 PAYLOADS:")
    print("==============================")

    for i, p in enumerate(points[:10], 1):
        print(f"{i}. {p.payload}")

    print("\n==============================")
    print("DONE")
    print("==============================\n")


if __name__ == "__main__":
    main()
