import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService } from '../../services/format.service';

@Pipe({ name: 'formatBps', pure: true })
export class FormatBpsPipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: number | null | undefined): string {
    return this.fmt.formatBps(value);
  }
}
