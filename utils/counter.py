def count_vehicles(results, model, line_y, valid_labels, seen_ids):
    total = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in valid_labels:
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            y1 = int(box.xyxy[0][1])
            y2 = int(box.xyxy[0][3])
            center_y = (y1 + y2) // 2

            if center_y > line_y and track_id not in seen_ids:
                seen_ids.add(track_id)
                total += 1

    return total