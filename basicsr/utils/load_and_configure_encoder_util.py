import torch
def load_and_configure_encoder(model, opt, logger):
    """
    Load weights for specified modules and selectively freeze them.

    Args:
        model (nn.Module): The initialized model.
        opt (dict): Dictionary of options including paths and model configurations.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        nn.Module: The updated model with modules loaded and frozen.
    """
    # 提取配置选项
    pretrained_weights = opt['Pretrained_encoder'].get('pretrained_weights', None)
    freeze_modules = opt['Pretrained_encoder'].get('freeze_modules', [])

    # 加载权重并冻结指定模块
    if freeze_modules:
        logger.info(f"指定要冻结的模块: {freeze_modules}")
        
        # 加载所有指定模块的预训练权重
        if pretrained_weights:
            try:
                checkpoint = torch.load(pretrained_weights)
                state_dict = checkpoint.get('params', checkpoint)
                
                # 为每个需要冻结的模块加载权重
                for module_name in freeze_modules:
                    # 使用原始模块名称，不转换为小写
                    # 提取该模块的权重
                    module_state_dict = {
                        k.replace(f'{module_name}.', ''): v 
                        for k, v in state_dict.items() 
                        if k.startswith(module_name)
                    }
                    
                    if module_state_dict:
                        # 获取模块实例，使用原始模块名称
                        module = getattr(model, module_name, None)
                        if module is not None:
                            missing_keys, unexpected_keys = module.load_state_dict(module_state_dict, strict=False)
                            if missing_keys or unexpected_keys:
                                logger.warning(f"{module_name} loading - Missing keys: {missing_keys}")
                                logger.warning(f"{module_name} loading - Unexpected keys: {unexpected_keys}")
                            logger.info(f"Loaded {module_name} weights")
                        else:
                            logger.warning(f"Module {module_name} not found in model")
            
            except Exception as e:
                logger.error(f"加载模块权重时出错: {str(e)}")

        # 计算模型总参数量
        total_model_params = sum(p.numel() for p in model.parameters())
        
        # 冻结指定的模块参数
        for module_name in freeze_modules:
            try:
                module = getattr(model, module_name, None)
                if module is not None:
                    total_params = sum(p.numel() for p in module.parameters())
                    frozen_params = 0
                    for param in module.parameters():
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_params += param.numel()
                    if frozen_params > 0:
                        percentage = (frozen_params / total_model_params) * 100
                        logger.info(f"已冻结 {module_name} 模块: {frozen_params:,} 个参数 (占模型总参数的 {percentage:.2f}%)")
                    else:
                        percentage = (total_params / total_model_params) * 100
                        logger.warning(f"{module_name} 模块没有可训练的参数（总参数量：{total_params:,}，占比 {percentage:.2f}%）")
                else:
                    logger.warning(f"未找到模块 {module_name}，跳过冻结操作")
            except Exception as e:
                logger.error(f"冻结模块 {module_name} 时发生错误: {str(e)}")

    return model
