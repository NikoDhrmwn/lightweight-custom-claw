/**
 * LiteClaw — Skills Management Tool
 *
 * Allows the agent to list, view, create, and refine skills on the fly,
 * enabling autonomous learning and procedural memory accumulation.
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import { loadSkillCatalog, getSkillByName, saveSkill } from '../core/skills.js';
import { getMemoryStore } from '../core/memory.js';

toolRegistry.register({
  name: 'manage_skills',
  description: 'Manage the agent skill catalog. List available skills, inspect skill instructions, or author/improve a skill dynamically when you discover a repeatable workflow.',
  category: 'utility',
  parameters: [
    {
      name: 'action',
      type: 'string',
      description: 'Operation: "list" (show all skills), "view" (read skill details), "create" (author new skill), or "update" (refine an existing skill)',
      required: true,
      enum: ['list', 'view', 'create', 'update'],
    },
    {
      name: 'name',
      type: 'string',
      description: 'The name / identifier of the skill (e.g. "ffmpeg-tools", "react-migration")',
      required: false,
    },
    {
      name: 'description',
      type: 'string',
      description: 'Short summary of what this skill does and when the agent should select it',
      required: false,
    },
    {
      name: 'body',
      type: 'string',
      description: 'The full markdown instructions, tips, and workflow steps for this skill',
      required: false,
    },
  ],
  usageNotes: [
    'Create a skill when you solve a novel, complex task with multiple steps that the user might need again.',
    'Update an existing skill when instructions fail or when you find a cleaner, more reliable way to do something.',
    'Use list or view to inspect instructions before performing specialized tasks.',
  ],
  examples: [
    {
      userIntent: 'save our custom ffmpeg workflow as a skill',
      arguments: {
        action: 'create',
        name: 'video-slicing',
        description: 'Instructions for extracting frames, creating storyboards, and re-encoding video via ffmpeg',
        body: '## Video Slicing with FFmpeg\n1. Check format with ffprobe\n2. Extract keyframes without transcode using -ss before -i',
      },
    },
    {
      userIntent: 'list active skills',
      arguments: { action: 'list' },
    },
  ],
  keywords: ['skills', 'skill', 'learn', 'knowledge', 'improve', 'save', 'workflow', 'procedural'],
  handler: async (args, context): Promise<ToolResult> => {
    const action = String(args.action ?? 'list').toLowerCase();
    const memory = getMemoryStore();

    try {
      if (action === 'list') {
        const skills = loadSkillCatalog();
        const topStats = memory.getTopSkills(10);
        const topMap = new Map(topStats.map(s => [s.skillName.toLowerCase(), s.count]));

        if (skills.length === 0) {
          return { success: true, output: 'No skills found in the catalog.' };
        }

        const formatted = skills.map(s => {
          const uses = topMap.get(s.name.toLowerCase()) ?? 0;
          return `• **${s.name}** — ${s.description} ${uses > 0 ? `(Used ${uses} times)` : ''}`;
        }).join('\n');

        return {
          success: true,
          output: `📚 **Skill Catalog (${skills.length} skills)**:\n\n${formatted}`,
        };
      }

      const name = String(args.name ?? '').trim();

      if (action === 'view') {
        if (!name) return { success: false, output: 'Missing skill "name".' };
        const skill = getSkillByName(name);
        if (!skill) {
          return { success: false, output: `Skill "${name}" not found in catalog.` };
        }

        memory.recordSkillUsage(skill.name, context?.sessionKey ?? 'default', 'view');

        return {
          success: true,
          output: `# Skill: ${skill.name}\n**Description**: ${skill.description}\n**Path**: \`${skill.path}\`\n\n${skill.body}`,
        };
      }

      if (action === 'create' || action === 'update') {
        if (!name) return { success: false, output: 'Missing skill "name".' };
        const desc = String(args.description ?? '').trim();
        const body = String(args.body ?? '').trim();

        if (!body) return { success: false, output: 'Missing skill "body" (markdown instructions).' };

        const existing = getSkillByName(name);
        if (action === 'create' && existing) {
          return {
            success: false,
            output: `Skill "${name}" already exists. Use action: "update" to modify it or choose another name.`,
          };
        }

        const savedPath = saveSkill(name, desc || existing?.description || `Skill for ${name}`, body);
        memory.recordSkillUsage(name, context?.sessionKey ?? 'default', action);

        return {
          success: true,
          output: `Successfully ${action === 'create' ? 'created' : 'updated'} skill "${name}". Saved to: \`${savedPath}\`.`,
        };
      }

      return { success: false, output: `Unknown action: "${action}".` };
    } catch (err: any) {
      return { success: false, output: `Skill error: ${err.message}` };
    }
  },
});
