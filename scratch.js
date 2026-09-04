async function test() {
  const url = 'https://klipy.com/gifs/shia-labeouf-clapping-6';
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)' }
    });
    const text = await res.text();
    const m = text.match(/<meta[^>]*property=["']og:video["'][^>]*content=["']([^"']+)["']/i) || 
              text.match(/<meta[^>]*content=["']([^"']+)["'][^>]*property=["']og:video["']/i) ||
              text.match(/<meta[^>]*property=["']og:image["'][^>]*content=["']([^"']+)["']/i) ||
              text.match(/<meta[^>]*content=["']([^"']+)["'][^>]*property=["']og:image["']/i);
    console.log(m ? m[1] : 'No og:video or og:image found');
  } catch (e) {
    console.error(e);
  }
}
test();
