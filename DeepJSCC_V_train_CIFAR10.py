import torch
import torch.nn as nn
import os
from tqdm import tqdm
import logging
from datetime import datetime, timedelta, timezone
import pytz
from utils import Load_cifar10_data, DatasetFolder
from models import ADJSCC_V

def setup_logger(work_dir):
    """Setup logger configuration"""
    # Create formatter with milliseconds
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    log_file = os.path.join(work_dir, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def setup_data():
    """Setup data loaders"""
    x_train, x_test = Load_cifar10_data()
    
    train_dataset = DatasetFolder(x_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    return train_loader, test_loader, len(test_dataset)

def train_epoch(model, train_loader, criterion, optimizer, epoch, logger):
    """Train for one epoch"""
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
    epoch_loss = 0
    num_batches = len(train_loader)

    for i, x_input in enumerate(pbar):
        x_input = x_input.cuda()
        
        SNR_TRAIN = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
        CR = 0.1 + 0.9 * torch.rand(x_input.shape[0], 1).cuda()
        
        x_rec = model(x_input, SNR_TRAIN, CR, CHANNEL)
        loss = criterion(x_input, x_rec).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        if i % PRINT_RREQ == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.4e}'
            })
    
    avg_epoch_loss = epoch_loss / num_batches
    # logger.info(f'Epoch {epoch} - Average Training Loss: {avg_epoch_loss:.4f}')
    return avg_epoch_loss


def evaluate(model, test_loader, criterion, test_dataset_size, logger):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for test_input in tqdm(test_loader, desc='Evaluating'):
            test_input = test_input.cuda()
            SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
            CR = 0.1 + 0.9 * torch.rand(test_input.shape[0], 1).cuda()
            
            test_rec = model(test_input, SNR_TEST, CR, CHANNEL)
            total_loss += criterion(test_input, test_rec).item() * test_input.size(0)
            
    avg_loss = total_loss / test_dataset_size
    # logger.info(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    tw = pytz.timezone('Asia/Taipei')
    current_time = datetime.now(tw)

    # Configuration
    config = {
        'BATCH_SIZE': 128,
        'EPOCHS': 400,
        'ESTOP': 60,
        'LEARNING_RATE': 1e-4,
        'PRINT_RREQ': 150,
        'CHANNEL': 'AWGN',
        'IMG_SIZE': [3, 32, 32],
        'N_channels': 256,
        'Kernel_sz': 5,
        'CONTINUE_TRAINING': False,
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Update global variables
    globals().update(config)

    work_dir = os.path.join(
        'results', 
        current_time.strftime('%Y%m%d_%H%M%S') + 
        f'_DeepJSCC_VLC_{Kernel_sz}x{Kernel_sz}_{CHANNEL}_{N_channels}_20'
    )
    os.makedirs(work_dir, exist_ok=True)
    logger = setup_logger(work_dir)
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    # Setup
    current_epoch = 0
    min_epoch = 0
    estop = 0
    enc_out_shape = [48, IMG_SIZE[1]//4, IMG_SIZE[2]//4]
    
    # Data loading
    train_loader, test_loader, test_dataset_size = setup_data()
    logger.info(f"Dataset loaded. Test dataset size: {test_dataset_size}")

    # Model initialization
    model = ADJSCC_V(enc_out_shape, Kernel_sz, N_channels).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load checkpoint if continuing training
    best_loss = 1e3
    if CONTINUE_TRAINING:
        checkpoint_path = f'./results/DeepJSCC_VLC_{Kernel_sz}x{Kernel_sz}_{CHANNEL}_{N_channels}_20/best.pth'
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        current_epoch = 204
        min_epoch = current_epoch
    
    # Training loop
    for epoch in range(current_epoch, EPOCHS):
        logger.info(f'\n{"="*20} Epoch {epoch+1} {"="*20}')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.4e}')
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch+1, logger)
        val_loss = evaluate(model, test_loader, criterion, test_dataset_size, logger)

        logger.info(f'Epoch {epoch+1} Summary:')
        logger.info(f'Train Loss: {train_loss:.6f}')
        logger.info(f'Val Loss: {val_loss:.6f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = work_dir + '/best.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, save_path)
            logger.info(f'New best model saved with loss: {best_loss:.6f}')
            min_epoch = epoch
            estop = 0
        else:
            estop += 1
        logger.info(f'Current epoch: {epoch+1}, Best epoch so far: {min_epoch+1}')
        
        if estop >= ESTOP:
            logger.info(f"Achieved early stop setting = {ESTOP}, bye bye")
            break

    logger.info('\nTraining Summary:')
    logger.info(f'Best Val Loss: {best_loss:.6f}')
    logger.info('Training completed!')

if __name__ == '__main__':
    main()