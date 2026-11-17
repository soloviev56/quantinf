-- Подготовка Markdown-файлов курса к сборке сайта (запускается до обработки Quarto).
-- 1. Если в документе нет title, ведущие заголовки первого уровня становятся
--    заголовком страницы (исходные .md остаются пригодными и для pandoc).
-- 2. Если рядом с лекцией лежит примеры.py, в начало страницы добавляется
--    ссылка на скачивание.

local function file_exists(path)
  local f = io.open(path, "r")
  if f then f:close() return true end
  return false
end

function Pandoc(doc)
  if not doc.meta.title then
    local parts = {}
    while #doc.blocks > 0 and doc.blocks[1].t == "Header" and doc.blocks[1].level == 1 do
      table.insert(parts, doc.blocks[1].content)
      doc.blocks:remove(1)
    end
    if #parts > 0 then
      local title = pandoc.Inlines({})
      for i, p in ipairs(parts) do
        if i > 1 then title:insert(pandoc.Space()) end
        title:extend(p)
      end
      doc.meta.title = title
      doc.meta.pagetitle = pandoc.utils.stringify(title)
    end
  end

  local input = quarto.doc.input_file
  local dir = pandoc.path.directory(input)
  if file_exists(pandoc.path.join({dir, "примеры.py"})) then
    local note = pandoc.Div({
      pandoc.Para({
        pandoc.Strong({pandoc.Str("Код к лекции:")}), pandoc.Space(),
        pandoc.Link({pandoc.Code("примеры.py")}, "примеры.py"),
        pandoc.Str(" — численная проверка утверждений лекции. Запуск: "),
        pandoc.Code("python примеры.py"),
        pandoc.Str(" (нужны NumPy и SciPy)."),
      })
    }, pandoc.Attr("", {"callout-note"}, {appearance = "simple"}))
    doc.blocks:insert(1, note)
  end
  return doc
end
