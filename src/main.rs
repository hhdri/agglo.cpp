use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

const VOCAB_SIZE: usize = 10_000;
const EMBEDDING_DIM: usize = 50;

fn load_glove(path: &str) -> Result<(Vec<String>, Vec<f32>), Box<dyn Error>> {
    // Buffered reading is much faster than calling `read_line` on File directly.
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Pre-allocate to avoid reallocations later.
    let mut vocab: Vec<String> = Vec::with_capacity(VOCAB_SIZE);
    let mut embeddings: Vec<f32> = vec![0.0; VOCAB_SIZE * EMBEDDING_DIM];

    for i in 0..VOCAB_SIZE {
        let mut line = String::new();
        // Stop early if the file runs out of lines.
        if reader.read_line(&mut line)? == 0 {
            break;
        }

        let mut parts = line.split_whitespace();
        // First token is the word.
        let word = parts
            .next()
            .ok_or("unexpected empty line while reading vocab")?;
        vocab.push(word.to_owned());

        // Read the EMBEDDING_DIM floats, accumulate L2-norm.
        let mut norm_sq = 0.0f32;
        for j in 0..EMBEDDING_DIM {
            let value: f32 = parts
                .next()
                .ok_or("unexpected end of line while reading embedding")?
                .parse()?;
            embeddings[i * EMBEDDING_DIM + j] = value;
            norm_sq += value * value;
        }

        // Normalise the vector in-place.
        let norm = norm_sq.sqrt();
        for j in 0..EMBEDDING_DIM {
            embeddings[i * EMBEDDING_DIM + j] /= norm;
        }
    }

    Ok((vocab, embeddings))
}

#[derive(Debug, PartialEq)]
struct PairSim {
    left: usize,
    right: usize,
    sim: f32
}

impl PairSim {
    fn new(left: usize, right: usize, sim: f32) -> PairSim {
        Self { left:std::cmp::min(left, right), right:std::cmp::max(left, right), sim }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (vocab, embeddings) = load_glove("./glove.6B.50d.txt")?;
    println!("Loaded {} words ({} floats)", vocab.len(), embeddings.len());
    println!("Word: {}, first: {},  last: {}", vocab[19], embeddings[19 * EMBEDDING_DIM], embeddings[19 * EMBEDDING_DIM + 49]);

    let mut row_dot_prods: Vec<f32> =  vec![0.0; VOCAB_SIZE];
    let mut most_sims = Vec::<Option<PairSim>>::new();
    for i in 0..VOCAB_SIZE {
        for j in 0..VOCAB_SIZE {
            let mut dot = 0.0;
            for k in 0..EMBEDDING_DIM {
                dot += embeddings[i * EMBEDDING_DIM + k] * embeddings[j * EMBEDDING_DIM + k];
            }
            row_dot_prods[j] = dot;
        }
        row_dot_prods[i] = f32::NEG_INFINITY;
        let (row_argmax_sim, &row_max_sim) = row_dot_prods
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        most_sims.push(Some(PairSim::new(i, row_argmax_sim, row_max_sim)));  // TODO: change this push to index assignment for multithreading to work
    }

    fn format_pair_sim(pair_sim: &PairSim, vocab: &Vec<String>) -> String{
        let result = format!("left: {:<15} right: {:<15} sim:{:>9.6}",  vocab[pair_sim.left], vocab[pair_sim.right], pair_sim.sim);
        result
    }

    for idx1 in 0..VOCAB_SIZE {
        let pair_sim1 = match &most_sims[idx1] {
            Some(pair_sim) => pair_sim,
            None => continue
        };
        let idx2= if pair_sim1.left != idx1 { pair_sim1.left } else { pair_sim1.right };
        let pair_sim2 = match &most_sims[idx2] {
            Some(pair_sim) => pair_sim,
            None => continue
        };

        if pair_sim1 != pair_sim2 {
            println!("{} {}", format_pair_sim(pair_sim1, &vocab), format_pair_sim(pair_sim2, &vocab));
        }

        if pair_sim1 ==  pair_sim2 || pair_sim1.sim >= pair_sim2.sim {
            most_sims[idx2] = None;
        } else {
            most_sims[idx1] = None;
        }
    }

    Ok(())
}
