import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService, FormattedReturn } from '../../services/format.service';

@Pipe({ name: 'formatReturn', pure: true })
export class FormatReturnPipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: number | null | undefined): FormattedReturn {
    return this.fmt.formatReturn(value);
  }
}
