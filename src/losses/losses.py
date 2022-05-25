import paddle.nn as nn
import paddle


class Ball_Detection_Loss(nn.Layer):
    def __init__(self, w, h, epsilon=1e-9):
        super(Ball_Detection_Loss, self).__init__()
        self.w = w
        self.h = h
        self.epsilon = epsilon

    def forward(self, pred_ball_position, target_ball_position):
        x_pred = pred_ball_position[:, :self.w]
        y_pred = pred_ball_position[:, self.w:]

        x_target = target_ball_position[:, :self.w]
        y_target = target_ball_position[:, self.w:]

        loss_ball_x = - paddle.mean(x_target * paddle.log(x_pred + self.epsilon) + (1 - x_target) * paddle.log(1 - x_pred + self.epsilon))
        loss_ball_y = - paddle.mean(y_target * paddle.log(y_pred + self.epsilon) + (1 - y_target) * paddle.log(1 - y_pred + self.epsilon))

        return loss_ball_x + loss_ball_y


class Events_Spotting_Loss(nn.Layer):
    def __init__(self, weights=(1, 3), num_events=2, epsilon=1e-9):
        super(Events_Spotting_Loss, self).__init__()
        self.weights = paddle.to_tensor(weights).reshape((1, 2))
        self.weights = self.weights / self.weights.sum()
        self.num_events = num_events
        self.epsilon = epsilon

    def forward(self, pred_events, target_events):
        self.weights = self.weights.cuda()
        return - paddle.mean(self.weights * (target_events * paddle.log(pred_events + self.epsilon) + (1. - target_events) * paddle.log(1 - pred_events + self.epsilon)))


class DICE_Smotth_Loss(nn.Layer):
    def __init__(self, epsilon=1e-9):
        super(DICE_Smotth_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_seg, target_seg):
        return 1. - ((paddle.sum(2 * pred_seg * target_seg) + self.epsilon) / (paddle.sum(pred_seg) + paddle.sum(target_seg) + self.epsilon))


class BCE_Loss(nn.Layer):
    def __init__(self, epsilon=1e-9):
        super(BCE_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_seg, target_seg):
        return - paddle.mean(target_seg * paddle.log(pred_seg + self.epsilon) + (1 - target_seg) * paddle.log(1 - pred_seg + self.epsilon))


class Segmentation_Loss(nn.Layer):
    def __init__(self, bce_weight=0.5):
        super(Segmentation_Loss, self).__init__()
        self.bce_criterion = BCE_Loss(epsilon=1e-9)
        self.dice_criterion = DICE_Smotth_Loss(epsilon=1e-9)
        self.bce_weight = bce_weight

    def forward(self, pred_seg, target_seg):
        target_seg = target_seg.cast('float32')
        loss_bce = self.bce_criterion(pred_seg, target_seg)
        loss_dice = self.dice_criterion(pred_seg, target_seg)
        loss_seg = (1 - self.bce_weight) * loss_dice + self.bce_weight * loss_bce
        return loss_seg
