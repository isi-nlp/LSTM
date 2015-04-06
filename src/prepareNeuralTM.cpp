#include <iostream>
#include <vector>
#include <queue>
#include <boost/unordered_map.hpp>
#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

using namespace std;
using namespace TCLAP;

#include "neuralLM.h" // for vocabulary
#include "util.h"

using namespace boost;
using namespace nplm;

void writeNgrams(const vector<vector<string> > &data, int source_context_size, int target_context_size, const vocabulary &input_vocab, int source_unk, const vocabulary &output_vocab, bool numberize, const string &filename)
{
    ofstream file(filename.c_str());
    if (!file)
    {
	cerr << "error: could not open " << filename << endl;
	exit(1);
    }

    int ngram_size = source_context_size + target_context_size + 1;

    // for each input and output line
    for (int i=0; i<data.size(); i++) {
        vector<int> nums;
	if (numberize) {
            for (int j=0; j<source_context_size; j++) {
	        nums.push_back(input_vocab.lookup_word(data[i][j], source_unk));
            }
            for (int j=source_context_size; j<ngram_size-1; j++) {
                nums.push_back(input_vocab.lookup_word(data[i][j]));
            }
	    nums.push_back(output_vocab.lookup_word(data[i][ngram_size-1]));
	} else {
            for (int j=0; j<ngram_size-1; j++) {
	        nums.push_back(lexical_cast<int>(data[i][j]));
            }
	    nums.push_back(lexical_cast<int>(data[i][ngram_size-1]));
	}
	for (int k=0; k<nums.size(); k++)
	  file << nums[k] << " ";
	file << endl;
    }
    file.close();
}
    
int main(int argc, char *argv[])
{
    int source_context_size, target_context_size, input_vocab_size, output_vocab_size, validation_size;
    bool numberize;
    string train_text, train_file, validation_text, validation_file, write_input_words_file, write_output_words_file, input_words_file, output_words_file;

    try
    {
	CmdLine cmd("Prepares training data for training a language model.", ' ', "0.1");

	// The options are printed in reverse order
    
    ValueArg<bool> arg_numberize("", "numberize", "If true, convert words to numbers. Default: true.", false, true, "bool", cmd);
    ValueArg<int> arg_input_vocab_size("", "input_vocab_size", "Vocabulary size.", false, -1, "int", cmd);
    ValueArg<int> arg_output_vocab_size("", "output_vocab_size", "Vocabulary size.", false, -1, "int", cmd);
    ValueArg<string> arg_input_words_file("", "input_words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
    ValueArg<string> arg_output_words_file("", "output_words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
    ValueArg<int> arg_source_context_size("", "source_context_size", "Size of input context.", true, -1, "int", cmd);
    ValueArg<int> arg_target_context_size("", "target_context_size", "Size of output context.", true, -1, "int", cmd);
	ValueArg<string> arg_write_input_words_file("", "write_input_words_file", "Output vocabulary.", false, "", "string", cmd);
	ValueArg<string> arg_write_output_words_file("", "write_output_words_file", "Output vocabulary.", false, "", "string", cmd);
    ValueArg<int> arg_validation_size("", "validation_size", "How many lines from training data to hold out for validation. Default: 0.", false, 0, "int", cmd);
	ValueArg<string> arg_validation_file("", "validation_file", "Output validation data (numberized n-grams).", false, "", "string", cmd);
	ValueArg<string> arg_validation_text("", "validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
	ValueArg<string> arg_train_file("", "train_file", "Output training data (numberized n-grams).", false, "", "string", cmd);
    ValueArg<string> arg_train_text("", "train_text", "Input training data (tokenized).", true, "", "string", cmd);

	cmd.parse(argc, argv);

	train_text = arg_train_text.getValue();
	train_file = arg_train_file.getValue();
	validation_file = arg_validation_file.getValue();
	validation_text = arg_validation_text.getValue();
	validation_size = arg_validation_size.getValue();
	write_input_words_file = arg_write_input_words_file.getValue();
	write_output_words_file = arg_write_output_words_file.getValue();
	source_context_size = arg_source_context_size.getValue();
	target_context_size = arg_target_context_size.getValue();
	input_vocab_size = arg_input_vocab_size.getValue();
	output_vocab_size = arg_output_vocab_size.getValue();
	input_words_file = arg_input_words_file.getValue();
	output_words_file = arg_output_words_file.getValue();
	numberize = arg_numberize.getValue();

    // check command line arguments

    // Notes:
    // - either --words_file or --vocab_size is required.
    // - if --words_file is set,
    // - if --vocab_size is not set, it is inferred from the length of the file
    // - if --vocab_size is set, it is an error if the vocab file has a different number of lines
    // - if --numberize 0 is set and --use_vocab f is not set, then the output model file will not have a vocabulary, and a warning should be printed.
    if ((input_words_file == "") && (input_vocab_size == -1)) {
        cerr << "Error: either --input_words_file or --input_vocab_size is required." << endl;
        exit(1);
    }
    if ((output_words_file == "") && (output_vocab_size == -1)) {
        cerr << "Error: either --output_words_file or --output_vocab_size is required." << endl;
        exit(1);
    }

    cerr << "Command line: " << endl;
    cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;
	
	const string sep(" Value: ");
	cerr << arg_train_text.getDescription() << sep << arg_train_text.getValue() << endl;
	cerr << arg_train_file.getDescription() << sep << arg_train_file.getValue() << endl;
	cerr << arg_validation_text.getDescription() << sep << arg_validation_text.getValue() << endl;
	cerr << arg_validation_file.getDescription() << sep << arg_validation_file.getValue() << endl;
	cerr << arg_validation_size.getDescription() << sep << arg_validation_size.getValue() << endl;
	cerr << arg_write_input_words_file.getDescription() << sep << arg_write_input_words_file.getValue() << endl;
	cerr << arg_write_output_words_file.getDescription() << sep << arg_write_output_words_file.getValue() << endl;
	cerr << arg_source_context_size.getDescription() << sep << arg_source_context_size.getValue() << endl;
	cerr << arg_target_context_size.getDescription() << sep << arg_target_context_size.getValue() << endl;
	cerr << arg_input_vocab_size.getDescription() << sep << arg_input_vocab_size.getValue() << endl;
	cerr << arg_output_vocab_size.getDescription() << sep << arg_output_vocab_size.getValue() << endl;
	cerr << arg_input_words_file.getDescription() << sep << arg_input_words_file.getValue() << endl;
	cerr << arg_output_words_file.getDescription() << sep << arg_output_words_file.getValue() << endl;
	cerr << arg_numberize.getDescription() << sep << arg_numberize.getValue() << endl;
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    string start(string("<s>")), stop(string("</s>"));

    // Read in input training data and validation data
    vector<vector<string> > train_data;
    readSentFile(train_text, train_data);
    
    vector<vector<string> > validation_data;
    if (validation_text != "") {
        readSentFile(validation_text, validation_data);
    }
    else if (validation_size > 0)
    {
        if (validation_size > train_data.size())
	{
	    cerr << "error: requested validation size is greater than training data size" << endl;
	    exit(1);
	}
	validation_data.insert(validation_data.end(), train_data.end() - validation_size, train_data.end());
	train_data.resize(train_data.size() - validation_size);
    }

    int ngram_size = source_context_size + target_context_size + 1;

    // Construct input vocabulary
    vocabulary input_vocab;
    int source_unk = input_vocab.insert_word("<source_unk>");
    int input_start = input_vocab.insert_word("<s>");
    int input_stop = input_vocab.insert_word("</s>");
    input_vocab.insert_word("<null>");

    // read input vocabulary from file
    if (input_words_file != "") {
        vector<string> words;
        readWordsFile(input_words_file, words);
        for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
            input_vocab.insert_word(*it);
        }
        // was input_vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
        if (input_vocab_size > 0) {
            if (input_vocab.size() != input_vocab_size) {
                cerr << "Error: size of input_vocabulary file " << input_vocab.size() << " != --input_vocab_size " << input_vocab_size << endl;
            }
        }
        // else, set it to the size of vocabulary read from file
        else {
            input_vocab_size = input_vocab.size();
        }
    }

    // or construct input vocabulary to contain top <input_vocab_size> most frequent words; all other words replaced by <unk>
    else {
        unordered_map<string,int> count;
        for (int i=0; i<train_data.size(); i++) {
            for (int j=0; j<ngram_size-1; j++) {
                count[train_data[i][j]] += 1; 
            }
        }

        input_vocab.insert_most_frequent(count, input_vocab_size);
        if (input_vocab.size() < input_vocab_size) {
            cerr << "warning: fewer than " << input_vocab_size << " types in training data; the unknown word will not be learned" << endl;
        }
    }

    // Construct output vocabulary
    vocabulary output_vocab;
    int output_start = output_vocab.insert_word("<s>");
    int output_stop = output_vocab.insert_word("</s>");

    // read output vocabulary from file
    if (output_words_file != "") {
        vector<string> words;
        readWordsFile(output_words_file, words);
        for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
            output_vocab.insert_word(*it);
        }
        // was output_vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
        if (output_vocab_size > 0) {
            if (output_vocab.size() != output_vocab_size) {
                cerr << "Error: size of output_vocabulary file " << output_vocab.size() << " != --output_vocab_size " << output_vocab_size << endl;
            }
        }
        // else, set it to the size of vocabulary read from file
        else {
            output_vocab_size = output_vocab.size();
        }
    }

    // or construct output vocabulary to contain top <output_vocab_size> most frequent words; all other words replaced by <unk>
    else {
        unordered_map<string,int> count;
        for (int i=0; i<train_data.size(); i++) {
	  count[train_data[i][ngram_size-1]] += 1; 
        }

        output_vocab.insert_most_frequent(count, output_vocab_size);
        if (output_vocab.size() < output_vocab_size) {
            cerr << "warning: fewer than " << output_vocab_size << " types in training data; the unknown word will not be learned" << endl;
        }
    }

    // write input vocabulary to file
    if (write_input_words_file != "") {
        cerr << "Writing vocabulary to " << write_input_words_file << endl;
        writeWordsFile(input_vocab.words(), write_input_words_file);
    }

    // write output vocabulary to file
    if (write_output_words_file != "") {
        cerr << "Writing vocabulary to " << write_output_words_file << endl;
        writeWordsFile(output_vocab.words(), write_output_words_file);
    }

    // Write out input and output numberized n-grams
    if (train_file != "")
    {
        cerr << "Writing training data to " << train_file << endl;
        writeNgrams(train_data, source_context_size, target_context_size, input_vocab, source_unk, output_vocab, numberize, train_file);

    }
    if (validation_file != "")
    {
        cerr << "Writing validation data to " << validation_file << endl;
        writeNgrams(validation_data, source_context_size, target_context_size, input_vocab, source_unk, output_vocab, numberize, validation_file);
    }
}
