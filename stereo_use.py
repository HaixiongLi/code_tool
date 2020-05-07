#评估估计的disp和ground truth的误差
def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()


#调整学习率
def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#warp操作
    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output

#build 2d cost volume
    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

#build 3d cost volume
    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1])
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()