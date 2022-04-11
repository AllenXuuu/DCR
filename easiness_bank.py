import torch



class EasinessBank:
    def __init__(self, config, dataset_size):
        self.config = config
        self.history_error = {}
        self.easiness = {}
        
        for i in range(dataset_size):
            self.history_error[i] = None
            self.easiness[i] = 1.
        
    def update_error(self, index, error ):
        assert len(index) == len(error)
        
        
        for i, e in zip(index,error):
            assert i in self.history_error and i in self.easiness
            his_e = self.history_error[i]
            self.history_error[i] = e.item()
            
            
            if his_e is None:
                self.easiness[i] = 1.
            else:
                dec = e / his_e
                dec = min(self.config.gamma_max,dec)
                dec = max(self.config.gamma_min,dec)
                self.easiness[i] *= dec
        
        
    def query_easiness(self,index):
        return torch.tensor([self.easiness[i] for i in index])
    
    
    
    
    def log(self,epoch,logger,writer):
        all_easiness = [a for a in self.easiness.values() if a is not None]
        all_error = [a for a in self.history_error.values() if a is not None]
        
        report = 'Bank Epoch %d' % epoch
        if len(all_easiness) > 0:
            min_easiness,max_easiness,mean_easiness = min(all_easiness), max(all_easiness), sum(all_easiness) / len(all_easiness)
            report = report + ' Easiness: mean %.2f min %.2f max %.2f' % (mean_easiness, min_easiness, max_easiness)
            if writer is not None:
                writer.add_scalar('Easiness/mean', mean_easiness, epoch)
                writer.add_scalar('Easiness/max', max_easiness, epoch)
                writer.add_scalar('Easiness/min', min_easiness, epoch)
        
        if len(all_error) > 0:
            min_err,max_err,mean_err = min(all_error), max(all_error), sum(all_error) / len(all_error)
            report = report + ' Error: mean %.2f min %.2f max %.2f' % (mean_err, min_err, max_err)
            if writer is not None:
                writer.add_scalar('Error/mean', min_err, epoch)
                writer.add_scalar('Error/max', max_err, epoch)
                writer.add_scalar('Error/min', mean_err, epoch)
        
        logger.info(report)
        