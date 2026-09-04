/**
 * LiteClaw — Kanban Task Board Tool
 *
 * Allows the agent to organize tasks, milestones, and project items on a persistent board.
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import { getMemoryStore, KanbanCard } from '../core/memory.js';

toolRegistry.register({
  name: 'manage_kanban',
  description: 'Manage persistent Kanban task boards. Create boards, add cards, move cards between columns (e.g. "todo", "in_progress", "done"), view board status, or complete tasks.',
  category: 'utility',
  parameters: [
    {
      name: 'action',
      type: 'string',
      description: 'Operation: "list_boards", "create_board", "list_cards", "add_card", "move_card", "delete_card"',
      required: true,
      enum: ['list_boards', 'create_board', 'list_cards', 'add_card', 'move_card', 'delete_card'],
    },
    {
      name: 'boardName',
      type: 'string',
      description: 'Name of the board to create or view (e.g. "Project Alpha", "Sprint 1")',
      required: false,
    },
    {
      name: 'boardId',
      type: 'string',
      description: 'The unique board identifier',
      required: false,
    },
    {
      name: 'cardId',
      type: 'string',
      description: 'The card identifier to move or delete',
      required: false,
    },
    {
      name: 'title',
      type: 'string',
      description: 'Title of the card to add',
      required: false,
    },
    {
      name: 'description',
      type: 'string',
      description: 'Description or checklist items for the card',
      required: false,
    },
    {
      name: 'column',
      type: 'string',
      description: 'Column name: "todo", "in_progress", "done", "blocked"',
      required: false,
    },
    {
      name: 'priority',
      type: 'string',
      description: 'Card priority: "low", "medium", "high", "urgent"',
      required: false,
      enum: ['low', 'medium', 'high', 'urgent'],
    },
    {
      name: 'dueDate',
      type: 'string',
      description: 'Optional due date or deadline string (e.g. "2026-09-10", "tomorrow 5pm")',
      required: false,
    },
  ],
  usageNotes: [
    'Use this to keep track of user goals, project tasks, and todos across chats.',
    'Common columns: "todo", "in_progress", "done".',
  ],
  examples: [
    { userIntent: 'create a kanban board for website redesign', arguments: { action: 'create_board', boardName: 'Website Redesign' } },
    { userIntent: 'add a task to setup authentication', arguments: { action: 'add_card', boardName: 'Website Redesign', title: 'Setup Auth with Supabase', column: 'todo', priority: 'high' } },
    { userIntent: 'view kanban board', arguments: { action: 'list_cards', boardName: 'Website Redesign' } },
  ],
  keywords: ['kanban', 'todo', 'task', 'board', 'cards', 'project', 'milestone', 'deadline'],
  handler: async (args, context): Promise<ToolResult> => {
    const action = String(args.action ?? 'list_boards').toLowerCase();
    const memory = getMemoryStore();
    const userKey = (context?.sessionKey ?? 'default').split(':')[0] || 'default';

    try {
      if (action === 'create_board') {
        const name = String(args.boardName ?? 'General Tasks').trim();
        const board = memory.createKanbanBoard(userKey, name);
        return {
          success: true,
          output: `Created Kanban board "${board.name}" (ID: \`${board.id}\`).`,
        };
      }

      if (action === 'list_boards') {
        const boards = memory.listKanbanBoards(userKey);
        if (boards.length === 0) {
          return {
            success: true,
            output: 'No Kanban boards found for this user. You can create one with action: "create_board".',
          };
        }
        const formatted = boards.map(b => `• **${b.name}** (ID: \`${b.id}\`, Created: ${new Date(b.createdAt).toLocaleDateString()})`).join('\n');
        return {
          success: true,
          output: `Kanban Boards (${boards.length}):\n${formatted}`,
        };
      }

      // Resolve boardId if boardName was provided
      let boardId = args.boardId ? String(args.boardId).trim() : '';
      if (!boardId && args.boardName) {
        const boards = memory.listKanbanBoards(userKey);
        const match = boards.find(b => b.name.toLowerCase() === String(args.boardName).trim().toLowerCase());
        if (match) {
          boardId = match.id;
        }
      }

      // If still no boardId, fallback to the most recent board or create a default one
      if (!boardId) {
        const boards = memory.listKanbanBoards(userKey);
        if (boards.length > 0) {
          boardId = boards[0].id;
        } else {
          const defaultBoard = memory.createKanbanBoard(userKey, 'General Tasks');
          boardId = defaultBoard.id;
        }
      }

      if (action === 'add_card') {
        const title = String(args.title ?? '').trim();
        if (!title) return { success: false, output: 'Missing card "title".' };
        const desc = String(args.description ?? '').trim();
        const column = String(args.column ?? 'todo').toLowerCase();
        const priority = (args.priority ?? 'medium') as KanbanCard['priority'];
        const dueDate = args.dueDate ? String(args.dueDate).trim() : undefined;

        const card = memory.addKanbanCard(boardId, title, desc, column, priority, dueDate);
        return {
          success: true,
          output: `Added card "${card.title}" to [${card.columnName.toUpperCase()}] in board \`${boardId}\` (Card ID: \`${card.id}\`).`,
        };
      }

      if (action === 'move_card') {
        const cardId = String(args.cardId ?? '').trim();
        if (!cardId) return { success: false, output: 'Missing "cardId".' };
        const column = String(args.column ?? 'done').toLowerCase();
        const ok = memory.moveKanbanCard(cardId, column);
        if (!ok) return { success: false, output: `Card \`${cardId}\` not found.` };
        return {
          success: true,
          output: `Moved card \`${cardId}\` to [${column.toUpperCase()}].`,
        };
      }

      if (action === 'delete_card') {
        const cardId = String(args.cardId ?? '').trim();
        if (!cardId) return { success: false, output: 'Missing "cardId".' };
        const ok = memory.deleteKanbanCard(cardId);
        if (!ok) return { success: false, output: `Card \`${cardId}\` not found.` };
        return {
          success: true,
          output: `Deleted card \`${cardId}\`.`,
        };
      }

      if (action === 'list_cards') {
        const cards = memory.listKanbanCards(boardId);
        if (cards.length === 0) {
          return {
            success: true,
            output: `Board \`${boardId}\` is currently empty.`,
          };
        }

        const columns = ['todo', 'in_progress', 'done', 'blocked'];
        const grouped: Record<string, KanbanCard[]> = {};
        for (const c of cards) {
          const col = c.columnName.toLowerCase();
          if (!grouped[col]) grouped[col] = [];
          grouped[col].push(c);
        }

        const parts: string[] = [`📋 **Kanban Board (${boardId})**\n`];
        for (const col of Object.keys(grouped)) {
          parts.push(`### [${col.toUpperCase()}] (${grouped[col].length})`);
          for (const card of grouped[col]) {
            const prio = card.priority ? `[${card.priority}]` : '';
            const due = card.dueDate ? `(Due: ${card.dueDate})` : '';
            parts.push(`• \`${card.id}\` **${card.title}** ${prio} ${due}`);
            if (card.description) {
              parts.push(`  ${card.description}`);
            }
          }
          parts.push('');
        }

        return {
          success: true,
          output: parts.join('\n'),
        };
      }

      return { success: false, output: `Unknown action: "${action}".` };
    } catch (err: any) {
      return { success: false, output: `Kanban error: ${err.message}` };
    }
  },
});
