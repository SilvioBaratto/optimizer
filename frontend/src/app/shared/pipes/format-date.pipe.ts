import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService } from '../../services/format.service';

@Pipe({ name: 'formatDate', pure: true })
export class FormatDatePipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: Date | string | null | undefined, format?: 'medium' | 'short' | 'iso'): string {
    return this.fmt.formatDate(value, format);
  }
}
