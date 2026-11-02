import OpenAI from 'openai';

/**
 * Where the scraper's model traffic goes, and which models it uses — decided once, here. Every
 * util and script calls ai() and names a rung from MODELS rather than building a client or naming a
 * model of its own.
 *
 * Traffic goes through Vercel AI Gateway rather than straight to OpenAI: one bill across the
 * products, scraper spend separable in gateway reporting, and a model change that costs an env var.
 * Still the OpenAI SDK, so the calls themselves didn't change.
 *
 * The base URL is an env var rather than a literal because it is the rollback: point it at another
 * OpenAI-compatible endpoint, set the key and the model names to match, restart.
 */

export const MODELS = {
  // Langcliffe parsing and replies, sector classification, reply handling, notification emails.
  // Was gpt-4o-mini; luna is two generations newer and cheaper on both axes.
  fast: process.env.MODEL_FAST ?? 'openai/gpt-6-luna',

  // Listing and buyer embeddings for Pinecone. UNCHANGED ON PURPOSE: the index holds 1,536-dim
  // vectors from this model, and a different embedding model would make every stored vector
  // meaningless — no error, just silently worse matches. Only the provider prefix is new.
  embed: process.env.MODEL_EMBED ?? 'openai/text-embedding-3-small',
};

let client = null;

/** The shared client, built on first use so a missing key fails the call that needed it. */
export function ai() {
  const apiKey = process.env.AI_GATEWAY_API_KEY;
  if (!apiKey) throw new Error('AI_GATEWAY_API_KEY env var is not set');
  if (!client) {
    client = new OpenAI({
      apiKey,
      baseURL: process.env.AI_GATEWAY_BASE_URL ?? 'https://ai-gateway.vercel.sh/v1',
      defaultHeaders: { 'ai-reporting-tags': 'service:acquiro-scraper' },
    });

    // Ask for embeddings as plain floats. Left unset, the SDK requests base64 and then decodes
    // the reply as base64 unconditionally — so if an endpoint ever answered with a float array
    // anyway, every vector would silently come back a quarter of its length and wrong. The
    // gateway documents float arrays and says nothing about base64, and these vectors go into a
    // Pinecone index that can't be repaired without a full rebuild, so float is the only format
    // worth trusting here. Set centrally so no call site can forget it.
    const createEmbedding = client.embeddings.create.bind(client.embeddings);
    client.embeddings.create = (body, options) => createEmbedding({ encoding_format: 'float', ...body }, options);
  }
  return client;
}
