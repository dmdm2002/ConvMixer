class LossDisPlayer(object):
    def __init__(self, name_list):
        self.count = 0
        self.name_list = name_list
        self.value_list = [0] * len(self.name_list)

    def record(self, vals):
        self.count += 1
        for i, value in enumerate(vals):
            if i == 0:
                self.value_list[i] += value.item()
            else:
                self.value_list[i] += value

    def get_avg_losses(self, full_dataset_length):
        avg_list = [0, 0]
        for i, totals in enumerate(self.value_list):
            if i == 0:
                avg_list[i] = totals / self.count
            else:
                avg_list[i] = totals / full_dataset_length

        return avg_list

    def display_value(self, do_display):
        if do_display:
            for i, total_loss in enumerate(self.value_list):
                avg_loss = total_loss / self.count
                print(f"{self.name_list[i]}: {avg_loss:.4f}   ", end="")

    def reset(self):
        self.count = 0
        self.value_list = [0] * len(self.name_list)