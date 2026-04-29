
def test_sharding(total_items, world_size):
    items = list(range(total_items))
    print(f"Total items: {total_items}, World size: {world_size}")

    all_shards = []
    for rank in range(world_size):
        shard = items[rank::world_size]
        all_shards.extend(shard)
        print(f"Rank {rank}: {shard}")

    assert sorted(all_shards) == items, "Sharding lost or duplicated items!"
    assert len(set(all_shards)) == len(all_shards), "Sharding has duplicates!"
    print("Verification: All items covered exactly once.\n")

test_sharding(13, 2)
test_sharding(10, 2)
test_sharding(10, 3)
test_sharding(5, 5)
