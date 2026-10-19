import MarkdownIt from "markdown-it";

// 回复是不可信文本：不接受内嵌 HTML，也不在渲染时加载图片资源。
const markdown = new MarkdownIt({ html: false, breaks: true });
const validateLink = markdown.validateLink;
markdown.validateLink = (url) => /^(https?:\/\/|mailto:)/i.test(url) && validateLink(url);
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  tokens[index].attrSet("target", "_blank");
  tokens[index].attrSet("rel", "noopener noreferrer");
  return renderer.renderToken(tokens, index, options);
};
markdown.renderer.rules.image = (tokens, index) => markdown.utils.escapeHtml(tokens[index].content);

export function renderReplyMarkdown(content) {
  return markdown.render(content);
}
