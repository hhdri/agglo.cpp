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

fn format_pair_sim(pair_sim: &PairSim, vocab: &Vec<String>) -> String{
    let result = format!("left: {:<15} right: {:<15} sim:{:>9.6}",  vocab[pair_sim.left], vocab[pair_sim.right], pair_sim.sim);
    result
}

impl Eq for PairSim {}
impl PartialOrd for PairSim {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.sim.partial_cmp(&self.sim)
    }
}

impl Ord for PairSim {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `unwrap` is safe because we ruled out NaN in `new`
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone)]
enum BTree {
    Leaf(usize),
    Node(Box<(BTree, BTree)>)
}

fn flatten_b_tree(tree: &BTree) -> Vec<usize> {
    let mut result = Vec::new();
    flatten_recursive(tree, &mut result);
    result
}

// Internal recursive helper function
fn flatten_recursive(tree: &BTree, result: &mut Vec<usize>) {
    match tree {
        BTree::Leaf(n) => result.push(*n),
        BTree::Node(node) => {
            flatten_recursive(&node.0, result);
            flatten_recursive(&node.1, result);
        }
    }
}

fn format_b_tree(tree: &BTree, vocab: &Vec<String>) -> String {
    match tree {
        BTree::Leaf(x) => {vocab[*x].clone()}
        BTree::Node(pair) => {
            let (left, right) = &**pair;
            format!("{{{}, {}}}",                     // recursion
                    format_b_tree(left,  vocab),
                    format_b_tree(right,  vocab))
        }
    }
}

fn cluster_step(old_clusters: &Vec<BTree>, original_embeddings: &Vec<f32>) -> Vec<BTree> {
    let n_old_clusters = old_clusters.len();

    let old_clusters_flat = old_clusters.iter()
        .map(flatten_b_tree);

    let mut embeddings = vec![0.0; EMBEDDING_DIM * old_clusters.len()];
    for (idx, items) in old_clusters_flat.enumerate() {
        let n_items_f32 = items.len() as f32;
        for item in items {
            for i in 0..EMBEDDING_DIM {
                embeddings[idx * EMBEDDING_DIM + i] += original_embeddings[item * EMBEDDING_DIM + i] / n_items_f32;
            }
        }
    }

    let mut row_dot_prods =  vec![0.0; n_old_clusters];
    let mut most_sims = Vec::<PairSim>::new();
    for i in 0..n_old_clusters {
        for j in 0..n_old_clusters {
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
        most_sims.push(PairSim::new(i, row_argmax_sim, row_max_sim));  // TODO: change this push to index assignment for multithreading to work
    }
    // Sort most_sims by similarity
    most_sims.sort_unstable();
    most_sims.dedup();

    let mut old_clusters_copy: Vec<Option<&BTree>> = old_clusters.into_iter().map(Some).collect();
    let mut new_clusters = Vec::<BTree>::new();

    for pair_sim in &most_sims {
        let leaf1 = match old_clusters_copy[pair_sim.left] {
            Some(b) => b.clone(),
            None => continue,
        };

        let leaf2 = match old_clusters_copy[pair_sim.right] {
            Some(b) => b.clone(),
            None => continue,
        };
        old_clusters_copy[pair_sim.left] = None;
        old_clusters_copy[pair_sim.right] = None;
        new_clusters.push(BTree::Node(Box::new((leaf1, leaf2))));
    }
    new_clusters.extend(old_clusters_copy.into_iter().map(|x|x.cloned()).flatten());
    
    new_clusters
}

fn main() -> Result<(), Box<dyn Error>> {
    let (vocab, embeddings) = load_glove("./glove.6B.50d.txt")?;

    let singleton_clusters: Vec<BTree> = (0..VOCAB_SIZE).map(BTree::Leaf).collect();
    println!("Starting clustering with {} singleton clusters", singleton_clusters.len());

    let mut prev_clusters = singleton_clusters.clone();
    for i in 0..27 {
        println!("Clustering step {}", i + 1);
        let new_clusters = cluster_step(&prev_clusters, &embeddings);
        println!("New clusters: {}", new_clusters.len());
        // for new_node in &new_clusters {
        //     println!("{}", format_b_tree(new_node, &vocab));
        // }
        prev_clusters = new_clusters;
    }
    Ok(())
}
