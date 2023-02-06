Tutorial
========
This guide can help you start with Lomas. For more detailed examples, see the Lomas repository on 
`GitHub <https://github.com/ZhiwenLiu99/lomas-tutorial/>`_. The following codes are the basic Python examples to show how to use
Lomas.

.. code-block:: python

  from lomas.preprocessor import Preprocessor
  from lomas.generator import Generator
  
  def main(config):
      # Preprocessing. define the file path and file name of the raw data
      data = Preprocessor(f_path=config['path'], 
                          f_name=config['filenames'], 
                          f_type=config['type'],
                          column_names=config['cols'])
      # Initialization. init Lomas model using the preprocessed data
      model = Generator(ip_id_dict=data.ip_id_dict, 
                         ordered_ippair=data.ordered_ippair, 
                         cdf_iat=data.cdf_iat, 
                         cdf_size=data.cdf_size)
      model.initialize(data.trace_input)
      # Hyperparameters setting. set the hyperparameters of Lomas model, and train the model
      model.train(num_topics=25, 
                  chunksize=2000, 
                  passes=20, 
                  iterations=400)
      # Generation. generate synthetic trace by iteratively sampling from Lomas model
      model.generate(time_limit=data.trace_input['ts'].max(), 
                     time_unit=config['ts_unit'])
  # Running 
  if __name__ == "__main__":
      main(config)

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorial_preprocessor
   tutorial_generator





