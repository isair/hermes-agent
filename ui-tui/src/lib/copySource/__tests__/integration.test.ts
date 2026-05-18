import { beforeEach, describe, expect, it } from 'vitest'

import { buildLineStartsFromRows, simpleOffsetFor } from '../offsetMaps.js'
import { evictMessage, getRange, listRanges, registerRange, resetRegistry } from '../registry.js'
import { toCopyText } from '../toCopyText.js'
import type { MsgSnapshot, SelectionPoint } from '../types.js'

/**
 * Integration tests: exercise the full transcript-virtual copy-source
 * pipeline end-to-end. Each test sets up a fake transcript by registering
 * one or more ranges, builds two SelectionPoints, and asserts the
 * resulting copy text is the byte-exact slice the user expects.
 *
 * These are the "design contract" tests from the rewrite plan. The unit
 * tests in offsetMaps.test.ts / toCopyText.test.ts exercise the building
 * blocks in isolation; these tests verify they compose correctly under
 * the real wiring pattern (one CopySource per block, per-block offset
 * maps, fence inner/outer registration).
 */

function registerWholeMsg(msgId: string, source: string, blockIndex = 0): number {
  const rows = source.split('\n')

  return registerRange({
    msgId,
    blockIndex,
    outerSource: source,
    visualLineCount: Math.max(1, rows.length),
    getOffset: simpleOffsetFor(source, buildLineStartsFromRows(rows))
  })
}

function registerFenceBlock(msgId: string, blockIndex: number, outerSource: string, innerSource: string): number {
  // innerOffset = position of inner content in outer (just past the opener line)
  const innerOffset = outerSource.indexOf(innerSource)
  expect(innerOffset).toBeGreaterThan(0) // sanity: inner should not start at 0

  const rows = outerSource.split('\n')

  return registerRange({
    msgId,
    blockIndex,
    outerSource,
    innerSource,
    innerOffset,
    visualLineCount: Math.max(1, rows.length),
    getOffset: simpleOffsetFor(outerSource, buildLineStartsFromRows(rows))
  })
}

const makeTranscript = (...ids: string[]): MsgSnapshot[] =>
  ids.map((id, order) => ({ id, order }))

beforeEach(() => {
  resetRegistry()
})

describe('integration: byte-exact copy text from selection', () => {
  it('whole-message selection emits the entire source', () => {
    const text = 'hello world\nsecond line'
    registerWholeMsg('m1', text)
    const transcript = makeTranscript('m1')
    const anchor: SelectionPoint = { kind: 'before-all' }
    const focus: SelectionPoint = { kind: 'after-all' }

    expect(toCopyText({ anchor, focus, transcript })).toBe(text)
  })

  it('selection spanning two messages joins their sources with a newline', () => {
    const m1Text = 'msg one line a\nmsg one line b'
    const m2Text = 'msg two line a'
    registerWholeMsg('m1', m1Text)
    registerWholeMsg('m2', m2Text)
    const transcript = makeTranscript('m1', 'm2')

    expect(
      toCopyText({
        anchor: { kind: 'before-all' },
        focus: { kind: 'after-all' },
        transcript
      })
    ).toBe(`${m1Text}\n${m2Text}`)
  })

  it('partial selection within a single range emits the inner slice', () => {
    const text = 'abcdefghij'
    const id = registerWholeMsg('m1', text)
    const transcript = makeTranscript('m1')

    // 'cdefg' spans cols [2..7) of the single visual line.
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 0, col: 2 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 0, col: 7 }

    expect(toCopyText({ anchor, focus, transcript })).toBe('cdefg')
  })

  it('fence-strip: both endpoints inside fence body yield bare code', () => {
    const outer = '```py\nprint("hello")\nprint("world")\n```'
    const inner = 'print("hello")\nprint("world")'
    const id = registerFenceBlock('m1', 1, outer, inner)
    const transcript = makeTranscript('m1')
    const range = getRange(id)!

    // Selection: from start of first inner line to end of second inner line.
    // visualLine 1 = first inner content row in the rendered fence (row 0
    // is the ```py opener), col 0 = first byte.
    const innerLine1Start = range.innerOffset
    const innerLine2End = range.innerOffset + inner.length

    // Build points that resolve to those exact source offsets:
    // visualLine 1 col 0 → offset = rowStart(1) = innerLine1Start (because
    // simpleOffsetFor with one row per source line gives rowStarts[1] =
    // length of row 0 + 1 = "```py".length + 1 = 6 = innerOffset).
    expect(innerLine1Start).toBe(6)

    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 1, col: 0 }
    // visualLine 2 col 14 → row 2 starts at offset 21, col 14 → 35 = innerLine2End.
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 2, col: 14 }

    expect(toCopyText({ anchor, focus, transcript })).toBe(inner)
    expect(toCopyText({ anchor, focus, transcript })).not.toContain('```')
  })

  it('fence: selection extending past the closer keeps the fence markers', () => {
    const outer = '```py\nprint("hello")\n```'
    const inner = 'print("hello")'
    const id = registerFenceBlock('m1', 1, outer, inner)
    const transcript = makeTranscript('m1')

    // Anchor at start of OPENER line (visualLine 0 col 0), focus past end.
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 0, col: 0 }
    const focus: SelectionPoint = { kind: 'after-all' }

    expect(toCopyText({ anchor, focus, transcript })).toBe(outer)
  })

  it('two messages, partial selection: anchor mid-msg1, focus mid-msg2', () => {
    const m1 = 'hello world'
    const m2 = 'second message'
    const id1 = registerWholeMsg('m1', m1)
    const id2 = registerWholeMsg('m2', m2)
    const transcript = makeTranscript('m1', 'm2')

    // Anchor: col 6 of m1 (start of "world").
    // Focus: col 6 of m2 (after "second").
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id1, visualLine: 0, col: 6 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id2, visualLine: 0, col: 6 }

    expect(toCopyText({ anchor, focus, transcript })).toBe('world\nsecond')
  })

  it('eviction: msg dropped from history → range gone → stale point gives empty', () => {
    const m1 = 'doomed msg'
    const id1 = registerWholeMsg('m1', m1)
    // Even with the transcript still listing m1, eviction wipes the range
    // from the registry. The selection point's rangeId no longer resolves.
    evictMessage('m1')
    const transcript = makeTranscript('m1')

    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id1, visualLine: 0, col: 0 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id1, visualLine: 0, col: 10 }

    expect(toCopyText({ anchor, focus, transcript })).toBe('')
    expect(listRanges()).toHaveLength(0)
  })

  it('re-registration preserves the rangeId (virtual-scroll unmount/remount)', () => {
    const text = 'abc'
    const id1 = registerWholeMsg('m1', text)
    const id2 = registerWholeMsg('m1', text)

    expect(id2).toBe(id1)
    expect(listRanges()).toHaveLength(1)
  })

  it('gap point between msgs slots correctly in document order', () => {
    const m1 = 'first'
    const m2 = 'second'
    const id1 = registerWholeMsg('m1', m1)
    const id2 = registerWholeMsg('m2', m2)
    const transcript = makeTranscript('m1', 'm2')

    // Gap between m1 (end) and m2 (start) — like clicking on a blank
    // spacer row. afterRangeId=id1 means the gap is AFTER range id1.
    // beforeRangeId=id2 means the gap is BEFORE range id2.
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id1, visualLine: 0, col: 0 }
    const focus: SelectionPoint = { kind: 'gap', afterRangeId: id1, beforeRangeId: id2 }

    // Should slice from col 0 of m1 to end of m1; m2 is past the gap.
    expect(toCopyText({ anchor, focus, transcript })).toBe('first')
  })

  it('gap → in-range covers everything from gap-before-range through the focus', () => {
    const m1 = 'first'
    const m2 = 'second'
    const id1 = registerWholeMsg('m1', m1)
    const id2 = registerWholeMsg('m2', m2)
    const transcript = makeTranscript('m1', 'm2')

    // Gap BEFORE m2 (so positioned right after m1's end). Focus mid-m2.
    const anchor: SelectionPoint = { kind: 'gap', afterRangeId: id1, beforeRangeId: id2 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id2, visualLine: 0, col: 3 }

    // Gap-after-m1 == position past m1's last visual line, so m1 isn't
    // included. Output is just the prefix of m2.
    expect(toCopyText({ anchor, focus, transcript })).toBe('sec')
  })

  it('reversed selection (focus before anchor) produces the same text', () => {
    const text = 'abcdefgh'
    const id = registerWholeMsg('m1', text)
    const transcript = makeTranscript('m1')

    const forward = toCopyText({
      anchor: { kind: 'in-range', rangeId: id, visualLine: 0, col: 1 },
      focus: { kind: 'in-range', rangeId: id, visualLine: 0, col: 5 },
      transcript
    })

    const reversed = toCopyText({
      anchor: { kind: 'in-range', rangeId: id, visualLine: 0, col: 5 },
      focus: { kind: 'in-range', rangeId: id, visualLine: 0, col: 1 },
      transcript
    })

    expect(forward).toBe('bcde')
    expect(reversed).toBe('bcde')
  })

  it('multi-block msg: per-block ranges concat correctly on full-msg selection', () => {
    // Simulate a markdown msg with three blocks: heading, paragraph, fence.
    const headingSrc = '# Title'
    const paraSrc = 'Some text with `inline` code.'
    const fenceOuter = '```js\nconst x = 1;\n```'
    const fenceInner = 'const x = 1;'

    registerRange({
      msgId: 'm1',
      blockIndex: 1,
      outerSource: headingSrc,
      visualLineCount: 1,
      getOffset: simpleOffsetFor(headingSrc, buildLineStartsFromRows([headingSrc]))
    })
    registerRange({
      msgId: 'm1',
      blockIndex: 2,
      outerSource: paraSrc,
      visualLineCount: 1,
      getOffset: simpleOffsetFor(paraSrc, buildLineStartsFromRows([paraSrc]))
    })
    const innerOffset = fenceOuter.indexOf(fenceInner)
    const fenceRows = fenceOuter.split('\n')
    registerRange({
      msgId: 'm1',
      blockIndex: 3,
      outerSource: fenceOuter,
      innerSource: fenceInner,
      innerOffset,
      visualLineCount: fenceRows.length,
      getOffset: simpleOffsetFor(fenceOuter, buildLineStartsFromRows(fenceRows))
    })
    const transcript = makeTranscript('m1')

    expect(
      toCopyText({
        anchor: { kind: 'before-all' },
        focus: { kind: 'after-all' },
        transcript
      })
    ).toBe(`${headingSrc}\n${paraSrc}\n${fenceOuter}`)
  })

  it('selection mid-paragraph through mid-next-paragraph: byte-exact across blocks', () => {
    const para1 = 'first paragraph'
    const para2 = 'second paragraph'

    const id1 = registerRange({
      msgId: 'm1',
      blockIndex: 1,
      outerSource: para1,
      visualLineCount: 1,
      getOffset: simpleOffsetFor(para1, buildLineStartsFromRows([para1]))
    })

    const id2 = registerRange({
      msgId: 'm1',
      blockIndex: 2,
      outerSource: para2,
      visualLineCount: 1,
      getOffset: simpleOffsetFor(para2, buildLineStartsFromRows([para2]))
    })

    const transcript = makeTranscript('m1')

    // Anchor: 6 chars into para1 (start of "paragraph").
    // Focus: 7 chars into para2 (after "second ").
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id1, visualLine: 0, col: 6 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id2, visualLine: 0, col: 7 }

    expect(toCopyText({ anchor, focus, transcript })).toBe('paragraph\nsecond ')
  })

  it('python fence: selecting a single code line via in-range points returns just that line', () => {
    // Regression: selecting the docstring line `    """packet says hi !!"""`
    // inside a python fence should copy exactly that line, not the
    // surrounding code or the whole fence.
    const fenceOuter = [
      '```python',
      'def greet(name: str) -> str:',
      '    """packet says hi !!"""',
      '    return f"awaaaaa hi {name} >w<"',
      '',
      'print(greet("ethie"))',
      '```'
    ].join('\n')

    const fenceLines = fenceOuter.split('\n')
    const innerSource = fenceLines.slice(1, -1).join('\n')
    const innerOffset = fenceLines[0]!.length + 1

    const id = registerRange({
      msgId: 'm1',
      blockIndex: 1,
      outerSource: fenceOuter,
      innerSource,
      innerOffset,
      visualLineCount: fenceLines.length,
      getOffset: simpleOffsetFor(fenceOuter, buildLineStartsFromRows(fenceLines))
    })

    const transcript = [{ id: 'm1', order: 0 }]

    // Visual row 2 = docstring line (row 0 = opener / chrome label,
    // row 1 = def line, row 2 = docstring). col 0 = line start, col
    // 27 = end of `    """packet says hi !!"""`.
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 2, col: 0 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 2, col: 27 }

    // Fence-stripping rule applies: both endpoints land in innerSource
    // bounds → output is the innerSource slice, not the outerSource
    // slice (which would include the line with no opener/closer
    // adjustment).
    expect(toCopyText({ anchor, focus, transcript })).toBe('    """packet says hi !!"""')
  })

  it('python fence: selecting one wrapped code line past visualLineCount clamps to last source row', () => {
    // What happens if the docstring is the LAST tracked source row and
    // the hit-test reports visualLine past visualLineCount (e.g.
    // because the renderer wrapped the line to multiple visual rows
    // but the block was registered with source-line-count only).
    //
    // Defensive fallback in pointToOffset clamps to last-row getOffset,
    // bounded by the line's source-end. So the slice is at MOST the
    // docstring line itself, never spilling into post-fence content.
    const fenceOuter = [
      '```python',
      '    """packet says hi !!"""',
      '```'
    ].join('\n')

    const fenceLines = fenceOuter.split('\n')
    const innerSource = fenceLines.slice(1, -1).join('\n')
    const innerOffset = fenceLines[0]!.length + 1

    const id = registerRange({
      msgId: 'm1',
      blockIndex: 1,
      outerSource: fenceOuter,
      innerSource,
      innerOffset,
      visualLineCount: fenceLines.length,
      getOffset: simpleOffsetFor(fenceOuter, buildLineStartsFromRows(fenceLines))
    })

    const transcript = [{ id: 'm1', order: 0 }]

    // visualLine=99 simulates a click past the tracked visual rows
    // (e.g. wrap-continuation row beyond visualLineCount). The
    // defensive clamp in pointToOffset defers to the last-row offset,
    // which gets clamped to the row's source-end by simpleOffsetFor.
    const anchor: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 1, col: 0 }
    const focus: SelectionPoint = { kind: 'in-range', rangeId: id, visualLine: 99, col: 0 }

    // Last tracked row is the closer line `\`\`\`` at visualLine=2,
    // not the docstring at visualLine=1. visualLine=99 clamps to
    // start of closer row. Slice covers docstring + trailing \n.
    // (Pre-fix this would have clamped to outerSource.length,
    // returning everything from the docstring to end of fence.)
    const result = toCopyText({ anchor, focus, transcript })
    // The fence-stripping rule requires BOTH points inside [innerOffset, innerEnd];
    // visualLine=99 → clamp to byte 38 (start of closer). innerEnd = 38
    // (innerOffset 10 + innerSource.length 28). So 38 <= 38 is at the
    // boundary — fence-stripping kicks in if `<= innerEnd`. Either way,
    // the result must NOT include the closer ``` line.
    expect(result).not.toContain('```')
    // And it must contain the docstring content.
    expect(result).toContain('packet says hi !!')
  })
})
